#include "rt_traversal.h"
#include "rt_op_unit.h"
#include "rt_unit.h"

#define BVH_WIDTH 4
#define RT_STACK_SIZE 5
#define BLAS_NODE_SIZE 160

#define RAY_TRANSFORM_LATENCY 6
#define RAY_BOX_INTERSECTION_LATENCY 24
#define RAY_TRI_INTERSECTION_LATENCY 16

using namespace vortex;

RestartTrailTraversal::RestartTrailTraversal(
    uint32_t tlas_ptr,
    uint32_t blas_ptr,
    uint32_t qBvh_ptr,
    uint32_t tri_ptr, 
    uint32_t tri_idx_ptr, 
    RTUnit* rt_unit
): 
    tlas_ptr(tlas_ptr),
    blas_ptr(blas_ptr),
    qBvh_ptr(qBvh_ptr),
    tri_ptr(tri_ptr), 
    tri_idx_ptr(tri_idx_ptr),
    traversal_stack(RT_STACK_SIZE),
    rt_unit_(rt_unit)
{}

void RestartTrailTraversal::traverse(RayBuffer &ray_buf, RtuTraceData* trace_data){
    level = 0;
    trail.fill(0);
    base_ptr = tlas_ptr;
    node_ptr = calcNodePtr(0);
    
    uint32_t blasIdx = 0;
    
    Ray ray = ray_buf.ray;
    BVHNode node;
    
    bool exit = false;
    
    while(!exit){

        read_node(&node);
        //std::cout << node.imask << " " << node.leafData << std::endl;
        if(!isLeaf(&node)){
            //Internal node
            // if(isTopLevel(&node)){
            //     std::cout << node.leftFirst << std::endl;
            // }
            
            std::vector<ChildIntersection> intersections;

            for(int i=0; i<BVH_WIDTH; i++){
                if(node.children[i].meta == 0) continue;
                float min_x = node.px + std::ldexp(float(node.children[i].qaabb[0]), node.ex);
                float min_y = node.py + std::ldexp(float(node.children[i].qaabb[1]), node.ey);
                float min_z = node.pz + std::ldexp(float(node.children[i].qaabb[2]), node.ez);

                float max_x = node.px + std::ldexp(float(node.children[i].qaabb[3]), node.ex);
                float max_y = node.py + std::ldexp(float(node.children[i].qaabb[4]), node.ey);
                float max_z = node.pz + std::ldexp(float(node.children[i].qaabb[5]), node.ez);

                float d = ray_box_intersect(isTopLevel(&node) ? ray_buf.ray : ray, min_x, min_y, min_z, max_x, max_y, max_z);
                //trace_data->pipeline_latency += RAY_BOX_INTERSECTION_LATENCY;

                if(d < ray_buf.hit.dist){
                    intersections.emplace_back(d, i);
                }
            }

            std::sort(intersections.begin(), intersections.end(), [](const ChildIntersection &a, const ChildIntersection &b) {
                return a.dist > b.dist; //farthest ------> closest
            });

            uint32_t k = trail[level];
            uint32_t dropCount = (k == BVH_WIDTH) ? intersections.size() - 1 : k;
            for(int i=0; i<dropCount; i++){
                if(intersections.size() > 0){
                    intersections.pop_back();
                }
            }
            
            if(intersections.size() == 0){
                exit = pop();
            }else{
                ChildIntersection closest = intersections.back();
                intersections.pop_back();

                uint32_t nodeIdx = node.leftFirst + closest.childIdx;
                node_ptr = calcNodePtr(nodeIdx);
                
                if(intersections.size() == 0){
                    trail[level] = BVH_WIDTH;
                }else{
                    for(auto iter = intersections.begin(); iter != intersections.end(); iter++){
                        nodeIdx = node.leftFirst + (*iter).childIdx;
                        push(
                            StackEntry(
                                calcNodePtr(nodeIdx), 
                                iter == intersections.begin()
                            )
                        );
                    }
                }
                level++;
            }

        }else{
            //Leaf Node
            if(isTopLevel(&node)){
                blasIdx = node.leafData;
                //std::cout << blasIdx << std::endl;
                uint32_t bvh_offset;
                uint32_t blas_node_ptr = blas_ptr + blasIdx * BLAS_NODE_SIZE;
                dcache_read(&bvh_offset, blas_node_ptr + 32 * sizeof(float), sizeof(uint32_t));

                float M[16];
                dcache_read(&M[0], blas_node_ptr + 16 * sizeof(float), 16 * sizeof(float));
                ray = ray_transform(ray_buf.ray, M);
                //trace_data->pipeline_latency += RAY_TRANSFORM_LATENCY;
                //std::cout << bvh_offset << std::endl;
                base_ptr = qBvh_ptr + bvh_offset * sizeof(BVHNode);
                node_ptr = base_ptr;
            }else{
                uint32_t triCount = node.leafData;
                uint32_t leftFirst = node.leftFirst + (blasIdx == 0 ? 0 : 1024); //fix!!!!!!!

                for (uint32_t i = 0; i < triCount; ++i) {
                    uint32_t triIdx = leftFirst + i;
                    //dcache_read(&triIdx, tri_idx_ptr + (leftFirst + i) * sizeof(uint32_t), sizeof(uint32_t));
                    
                    uint32_t tri_addr = tri_ptr + triIdx * sizeof(Triangle);

                    Triangle tri;
                    dcache_read(&tri, tri_addr, sizeof(Triangle));

                    float bx, by, bz;
                    float d = ray_tri_intersect(ray, tri, bx, by, bz);
                    //trace_data->pipeline_latency += RAY_TRI_INTERSECTION_LATENCY;

                    if (d < ray_buf.hit.dist) {
                        ray_buf.hit.dist = d;
                        ray_buf.hit.bx = bx;
                        ray_buf.hit.by = by;
                        ray_buf.hit.bz = bz;
                        ray_buf.hit.blasIdx = blasIdx;
                        ray_buf.hit.triIdx = triIdx;
                    }
                }

                exit = pop();
            }
        }
    }
}


void RestartTrailTraversal::push(StackEntry e){
    traversal_stack.push(e);
}

int32_t RestartTrailTraversal::findNextParentLevel(){
    for(int i=level-1; i>=0; i--){
        if(trail[i] != BVH_WIDTH){
            return i;
        }
    }
    return -1;
}

bool RestartTrailTraversal::pop(){
    int32_t parentLevel = findNextParentLevel();

    if(parentLevel < 0){
        return true;
    }

    trail[parentLevel]++;

    for(int i=parentLevel+1; i<MAX_LEVEL; i++){
        trail[i] = 0;
    }

    if(traversal_stack.empty()){
        base_ptr = tlas_ptr;
        node_ptr = tlas_ptr;
        level = 0;
        //std::cout << "Restarting..." << std::endl;
    }else{
        StackEntry e = traversal_stack.pop();
        node_ptr = e.node_ptr;
        if(e.last){
            trail[parentLevel] = BVH_WIDTH;
        }

        level = parentLevel + 1;
    }
    return false;
}

void RestartTrailTraversal::read_node(BVHNode *node){
    dcache_read(node, node_ptr, sizeof(BVHNode));
}

bool RestartTrailTraversal::isTopLevel(BVHNode *node){
    return (uint32_t)(node->imask) == 1;
}

bool RestartTrailTraversal::isLeaf(BVHNode *node){
    return (isTopLevel(node) && node->leafData != UINT32_MAX) || (!isTopLevel(node) && node->leafData != 0);
}

void RestartTrailTraversal::dcache_read(void* data, uint64_t addr, uint32_t size) {
    rt_unit_->dcache_read(data, addr, size);
}