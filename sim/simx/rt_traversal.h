    
#pragma once
#include "rt_trace.h"
#include "dcrs.h"
#include "types.h"
#include <array>

#define LARGE_FLOAT 1e30f
#define MAX_TRAIL_LEVEL 32

namespace vortex {

struct BVHChildData {
    uint8_t meta;
    uint8_t qaabb[6];
};

struct BVHNode {
    float px, py, pz;
    int8_t ex, ey, ez;

    // 00: tlas internal
    // 01: tlas leaf
    // 10: bvh internal
    // 11: bvh leaf
    uint8_t imask;

    uint32_t leftFirst; //First Child Idx
    uint32_t leafData; //blasIdx for TLAS, triCount for BVH
    
    BVHChildData children[RT_BVH_WIDTH];
};

struct BLASNode {
    uint32_t bvh_offset;
    float invTransform[12];
};

struct Triangle {
    float v0_x, v0_y, v0_z, v1_x, v1_y, v1_z, v2_x, v2_y, v2_z;
};

struct Ray {
    float ro_x, ro_y, ro_z, rd_x, rd_y, rd_z;
};

struct Hit {
    float t = LARGE_FLOAT;
    float u, v;
    uint32_t blasIdx;
    uint32_t triIdx;
};

struct TraversalStackEntry {
    uint32_t node_ptr;
    bool last;

    TraversalStackEntry() : node_ptr(0), last(false) {}
    TraversalStackEntry(uint32_t _node_ptr) : node_ptr(_node_ptr), last(false) {}
    TraversalStackEntry(uint32_t _node_ptr, bool _last) : node_ptr(_node_ptr), last(_last) {}
};

typedef ShortStack<TraversalStackEntry, RT_STACK_SIZE> TraversalStack;
typedef std::array<uint32_t, MAX_TRAIL_LEVEL> TraversalTrail; //trail[i]: 0 ~ BVH_WIDTH

struct TraversalState {
    Ray ray;
    Hit hit, best_hit;
    TraversalTrail trail;
    TraversalStack stack;
    uint32_t root_ptr;
    uint32_t root_level;
    uint32_t level;
    
    TraversalState(){}
    TraversalState(Ray ray, uint32_t root_ptr){
        this->ray = ray;
        this->hit = Hit();
        this->best_hit = Hit();
        this->trail = {};
        this->stack = TraversalStack();
        this->root_ptr = root_ptr;
        this->root_level = 0;
        this->level = 0;
    }

    int32_t findNextParentLevel(){
        for(int i=level-1; i>=root_level; i--){
            if(trail[i] != RT_BVH_WIDTH){
                return i;
            }
        }
        return -1;
    }

    uint32_t pop(uint32_t& node_ptr){
        int32_t parentLevel = findNextParentLevel();

        if(parentLevel < 0){
            return VX_RT_TRAVERSAL_STATUS_FINISHED;
        }

        trail[parentLevel]++;

        for(int i=parentLevel+1; i<MAX_TRAIL_LEVEL; i++){
            trail[i] = 0;
        }

        if(stack.empty()){
            return VX_RT_TRAVERSAL_STATUS_RESTART;
        }

        auto e = stack.pop();
        node_ptr = e.node_ptr;
        if(e.last){
            trail[parentLevel] = RT_BVH_WIDTH;
        }

        level = parentLevel + 1;
        return VX_RT_TRAVERSAL_STATUS_CONTINUE; 
    }
};

class RTUnit;
class BVHTraverser{
    public:
        BVHTraverser(RTUnit* rt_unit, const DCRS &dcrs);

        uint32_t traverse(TraversalState& state, per_thread_info &thread_info);

        Ray ray_transform(const Ray &ray, float *transform_matrix);
    private:
        
        void read_node(BVHNode *node, uint32_t node_ptr);
        bool isTopLevel(BVHNode *node);
        bool isLeaf(BVHNode *node);
        uint32_t calcNodePtr(uint32_t root_ptr, uint32_t idx){ return root_ptr + idx * sizeof(BVHNode); }
        void dcache_read(void* data, uint64_t addr, uint32_t size);
        void dcache_write(const void* data, uint64_t addr, uint32_t size);
        
        float ray_tri_intersect(const Ray &ray, const Triangle &tri, float &u, float &v);
        float ray_box_intersect(const Ray &ray, float min_x, float min_y, float min_z, float max_x, float max_y, float max_z);

        uint32_t tlas_ptr, blas_ptr, qBvh_ptr, tri_ptr, tri_idx_ptr;

        const DCRS& dcrs_;
        RTUnit* rt_unit_;
};
}