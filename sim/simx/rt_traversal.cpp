#include "rt_traversal.h"
#include "rt_op_unit.h"
#include "rt_unit.h"

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
    qBvh_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_QBVH_PTR);
    tri_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_PTR);
    tri_idx_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_IDX_PTR);

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
                        stack.push(calcNodePtr(base_ptr, nodeIdx), iter == intersections.begin());
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
        TraversalStack::Entry e = stack.pop();
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