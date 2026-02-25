#include "rt_traversal.h"
#include "rt_unit.h"

#include <algorithm>
#include <cmath> 

#define EPSILON 1e-6f
// #define RAY_TRANSFORM_LATENCY 6
// #define RAY_BOX_INTERSECTION_LATENCY 24
// #define RAY_TRI_INTERSECTION_LATENCY 16

using namespace vortex;

struct ChildIntersection {
    float dist;
    uint32_t childIdx;

    ChildIntersection(float _dist, uint32_t _childIdx)
        : dist(_dist)
        , childIdx(_childIdx)
    {}
};

BVHTraverser::BVHTraverser(RTUnit* rt_unit, const DCRS &dcrs): rt_unit_(rt_unit), dcrs_(dcrs){}

bool BVHTraverser::traverse(
    const Ray& ray, 
    Hit& hit, 
    TraversalTrail& trail, 
    TraversalStack& stack,
    per_thread_info &thread_info
){
    tlas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TLAS_PTR);
    blas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BLAS_PTR);
    qBvh_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BVH_PTR);
    tri_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_PTR);
    //tri_idx_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_IDX_PTR);

    uint32_t level = 0;
    //trail.fill(0);
    uint32_t base_ptr = tlas_ptr;
    uint32_t node_ptr = calcNodePtr(base_ptr, 0);
    
    uint32_t blasIdx = 0;
    
    Ray cur_ray = ray;
    BVHNode node;
    
    bool exit = false;
    
    while(!exit){

        read_node(&node, node_ptr);
        thread_info.RT_mem_accesses.emplace_back(node_ptr, sizeof(BVHNode),TransactionType::BVH_INTERNAL_NODE);

        if(!isLeaf(&node)){
            std::vector<ChildIntersection> intersections;

            for(int i=0; i<RT_BVH_WIDTH; i++){
                if(node.children[i].meta == 0) continue;
                float min_x = node.px + std::ldexp(float(node.children[i].qaabb[0]), node.ex);
                float min_y = node.py + std::ldexp(float(node.children[i].qaabb[1]), node.ey);
                float min_z = node.pz + std::ldexp(float(node.children[i].qaabb[2]), node.ez);

                float max_x = node.px + std::ldexp(float(node.children[i].qaabb[3]), node.ex);
                float max_y = node.py + std::ldexp(float(node.children[i].qaabb[4]), node.ey);
                float max_z = node.pz + std::ldexp(float(node.children[i].qaabb[5]), node.ez);

                float d = ray_box_intersect(isTopLevel(&node) ? ray : cur_ray, min_x, min_y, min_z, max_x, max_y, max_z);

                if(d < hit.dist){
                    intersections.emplace_back(d, i);
                }
            }

            std::sort(intersections.begin(), intersections.end(), [](const ChildIntersection &a, const ChildIntersection &b) {
                return a.dist > b.dist; //farthest ------> closest
            });

            uint32_t k = trail[level];
            uint32_t dropCount = (k == RT_BVH_WIDTH) ? intersections.size() - 1 : k;
            for(int i=0; i<dropCount; i++){
                if(intersections.size() > 0){
                    intersections.pop_back();
                }
            }
            
            if(intersections.size() == 0){
                exit = pop(base_ptr, node_ptr, level, trail, stack);
            }else{
                ChildIntersection closest = intersections.back();
                intersections.pop_back();

                uint32_t nodeIdx = node.leftFirst + closest.childIdx;
                node_ptr = calcNodePtr(base_ptr, nodeIdx);
                
                if(intersections.size() == 0){
                    trail[level] = RT_BVH_WIDTH;
                }else{
                    for(auto iter = intersections.begin(); iter != intersections.end(); iter++){
                        nodeIdx = node.leftFirst + (*iter).childIdx;
                        stack.push({calcNodePtr(base_ptr, nodeIdx), iter == intersections.begin()});
                    }
                }
                level++;
            }

        }else{
            //Leaf Node
            if(isTopLevel(&node)){
                blasIdx = node.leafData;
                uint32_t blas_node_ptr = blas_ptr + blasIdx * 160;
                
                BLASNode blas_node;
                dcache_read(&blas_node, blas_node_ptr, sizeof(BLASNode));
                thread_info.RT_mem_accesses.emplace_back(blas_node_ptr, sizeof(BLASNode),TransactionType::BVH_INSTANCE_LEAF);

                cur_ray = ray_transform(ray, blas_node.invTransform);

                base_ptr = qBvh_ptr + blas_node.bvh_offset * sizeof(BVHNode);
                node_ptr = base_ptr;
            }else{
                uint32_t triCount = node.leafData;
                uint32_t leftFirst = node.leftFirst;

                for (uint32_t i = 0; i < triCount; ++i) {
                    uint32_t triIdx = leftFirst + i;
                    //dcache_read(&triIdx, tri_idx_ptr + (leftFirst + i) * sizeof(uint32_t), sizeof(uint32_t));
                    
                    uint32_t tri_addr = tri_ptr + triIdx * sizeof(Triangle);

                    Triangle tri;
                    dcache_read(&tri, tri_addr, sizeof(Triangle));
                    

                    float bx, by, bz;
                    float d = ray_tri_intersect(cur_ray, tri, bx, by, bz);

                    if (d < hit.dist) {
                        hit.pending_dist = d;
                        //hit.dist = d;
                        hit.bx = bx;
                        hit.by = by;
                        hit.bz = bz;
                        hit.blasIdx = blasIdx;
                        hit.triIdx = triIdx;
                        
                        thread_info.RT_mem_accesses.emplace_back(tri_addr, sizeof(Triangle),TransactionType::BVH_QUAD_LEAF_HIT);

                        //-------clear stack for now to ensure correctness--------
                        while(!stack.empty()){
                            stack.pop();
                        }
                        //--------------------------------------------------------
                        
                        return false;
                    }else{
                        thread_info.RT_mem_accesses.emplace_back(tri_addr, sizeof(Triangle),TransactionType::BVH_QUAD_LEAF);
                    }
                }

                exit = pop(base_ptr, node_ptr, level, trail, stack);
            }
        }
    }

    return true;
}

int32_t BVHTraverser::findNextParentLevel(const uint32_t level, const TraversalTrail& trail){
    for(int i=level-1; i>=0; i--){
        if(trail[i] != RT_BVH_WIDTH){
            return i;
        }
    }
    return -1;
}

bool BVHTraverser::pop(
    uint32_t& base_ptr,
    uint32_t& node_ptr,
    uint32_t& level,
    TraversalTrail& trail,
    TraversalStack& stack
){
    int32_t parentLevel = findNextParentLevel(level, trail);

    if(parentLevel < 0){
        return true;
    }

    trail[parentLevel]++;

    for(int i=parentLevel+1; i<MAX_TRAIL_LEVEL; i++){
        trail[i] = 0;
    }

    if(stack.empty()){
        base_ptr = tlas_ptr;
        node_ptr = tlas_ptr;
        level = 0;
        //std::cout << "Restarting..." << std::endl;
    }else{
        auto e = stack.pop();
        node_ptr = e.node_ptr;
        if(e.last){
            trail[parentLevel] = RT_BVH_WIDTH;
        }

        level = parentLevel + 1;
    }
    return false;
}

void BVHTraverser::read_node(BVHNode *node, uint32_t node_ptr){
    dcache_read(node, node_ptr, sizeof(BVHNode));
}

bool BVHTraverser::isTopLevel(BVHNode *node){
    return (uint32_t)(node->imask) == 1;
}

bool BVHTraverser::isLeaf(BVHNode *node){
    return (isTopLevel(node) && node->leafData != UINT32_MAX) || (!isTopLevel(node) && node->leafData != 0);
}

void BVHTraverser::dcache_read(void* data, uint64_t addr, uint32_t size) {
    rt_unit_->dcache_read(data, addr, size);
}

Ray BVHTraverser::ray_transform(const Ray &ray, float *transform_matrix){
    float m00 = transform_matrix[0];
    float m01 = transform_matrix[1];
    float m02 = transform_matrix[2];
    float m03 = transform_matrix[3];

    float m10 = transform_matrix[4];
    float m11 = transform_matrix[5];
    float m12 = transform_matrix[6];
    float m13 = transform_matrix[7];
    
    float m20 = transform_matrix[8];
    float m21 = transform_matrix[9];
    float m22 = transform_matrix[10];
    float m23 = transform_matrix[11];

    // float m30 = transform_matrix[12];
    // float m31 = transform_matrix[13];
    // float m32 = transform_matrix[14];
    // float m33 = transform_matrix[15];

    Ray transformed_ray;
    transformed_ray.ro_x = m00 * ray.ro_x + m01 * ray.ro_y + m02 * ray.ro_z + m03;
    transformed_ray.ro_y = m10 * ray.ro_x + m11 * ray.ro_y + m12 * ray.ro_z + m13;
    transformed_ray.ro_z = m20 * ray.ro_x + m21 * ray.ro_y + m22 * ray.ro_z + m23;

    transformed_ray.rd_x = m00 * ray.rd_x + m01 * ray.rd_y + m02 * ray.rd_z;
    transformed_ray.rd_y = m10 * ray.rd_x + m11 * ray.rd_y + m12 * ray.rd_z;
    transformed_ray.rd_z = m20 * ray.rd_x + m21 * ray.rd_y + m22 * ray.rd_z;
    return transformed_ray;
}

float BVHTraverser::ray_tri_intersect(const Ray &ray, const Triangle &tri, float &bx, float &by, float &bz){
    float v0_x = tri.v0_x, v0_y = tri.v0_y, v0_z = tri.v0_z;
    float v1_x = tri.v1_x, v1_y = tri.v1_y, v1_z = tri.v1_z;
    float v2_x = tri.v2_x, v2_y = tri.v2_y, v2_z = tri.v2_z;

    float ro_x = ray.ro_x, ro_y = ray.ro_y, ro_z = ray.ro_z;
    float rd_x = ray.rd_x, rd_y = ray.rd_y, rd_z = ray.rd_z;
    
    float edge1_x = v1_x - v0_x;
    float edge1_y = v1_y - v0_y;
    float edge1_z = v1_z - v0_z;

    float edge2_x = v2_x - v0_x;
    float edge2_y = v2_y - v0_y;
    float edge2_z = v2_z - v0_z;

    float h_x = rd_y * edge2_z - rd_z * edge2_y;
    float h_y = rd_z * edge2_x - rd_x * edge2_z;
    float h_z = rd_x * edge2_y - rd_y * edge2_x;

    float a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z;
    if (fabs(a) < EPSILON){
        return LARGE_FLOAT;
    }

    float f = 1 / a;
    float s_x = ro_x - v0_x;
    float s_y = ro_y - v0_y;
    float s_z = ro_z - v0_z;

    float w1 = f * (s_x * h_x + s_y * h_y + s_z * h_z);
    if (w1 < 0 || w1 > 1){
        return LARGE_FLOAT;
    }
        
    float q_x = s_y * edge1_z - s_z * edge1_y;
    float q_y = s_z * edge1_x - s_x * edge1_z;
    float q_z = s_x * edge1_y - s_y * edge1_x;

    const float w2 = f * (rd_x * q_x + rd_y * q_y + rd_z * q_z);
    if (w2 < 0 || w1 + w2 > 1){
        return LARGE_FLOAT;
    }
        
    const float tf = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z);
    if (tf <= EPSILON){
        return LARGE_FLOAT;
    }

    bx = w1;
    by = w2;
    bz = 1 - w1 - w2;
    return tf;
}

float BVHTraverser::ray_box_intersect(const Ray &ray, float min_x, float min_y, float min_z, float max_x, float max_y, float max_z){
    float ro_x = ray.ro_x, ro_y = ray.ro_y, ro_z = ray.ro_z;
    float rd_x = ray.rd_x, rd_y = ray.rd_y, rd_z = ray.rd_z;
    float idir_x, idir_y, idir_z, tmin, tmax, tx1, tx2, ty1, ty2, tz1, tz2;

    idir_x = 1.0f / rd_x;
    idir_y = 1.0f / rd_y;
    idir_z = 1.0f / rd_z;
    tx1 = (min_x - ro_x) * idir_x;
    tx2 = (max_x - ro_x) * idir_x;
    tmin = std::min(tx1, tx2);
    tmax = std::max(tx1, tx2);
    ty1 = (min_y - ro_y) * idir_y;
    ty2 = (max_y - ro_y) * idir_y;
    tmin = std::max(tmin, std::min(ty1, ty2));
    tmax = std::min(tmax, std::max(ty1, ty2));
    tz1 = (min_z - ro_z) * idir_z;
    tz2 = (max_z - ro_z) * idir_z;
    tmin = std::max(tmin, std::min(tz1, tz2));
    tmax = std::min(tmax, std::max(tz1, tz2));
    return tmax < tmin || tmax <= 0 ? LARGE_FLOAT : tmin;
}