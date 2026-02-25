    
#pragma once
#include "rt_trace.h"
#include "dcrs.h"
#include "types.h"
#include <array>

#define LARGE_FLOAT 1e30f
#define MAX_TRAIL_LEVEL 32
#define TRAVERSAL_STACK_CAPACITY 5

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
    float dist = LARGE_FLOAT, pending_dist;
    float bx, by, bz;
    uint32_t blasIdx, triIdx;
};

struct TraversalStackEntry {
    uint32_t node_ptr;
    bool last;

    TraversalStackEntry() : node_ptr(0), last(false) {}
    TraversalStackEntry(uint32_t _node_ptr, bool _last) : node_ptr(_node_ptr), last(_last) {}
};

typedef ShortStack<TraversalStackEntry, TRAVERSAL_STACK_CAPACITY> TraversalStack;
typedef std::array<uint32_t, MAX_TRAIL_LEVEL> TraversalTrail; //trail[i]: 0 ~ BVH_WIDTH

class RTUnit;
class BVHTraverser{
    public:
        BVHTraverser(RTUnit* rt_unit, const DCRS &dcrs);
        bool traverse(const Ray& ray, Hit& hit, TraversalTrail& trail, TraversalStack& traversal_stack, per_thread_info &thread_info);
    private:
        bool pop(uint32_t& base_ptr, uint32_t& node_ptr, uint32_t& level, TraversalTrail& trail, TraversalStack& traversal_stack);
        int32_t findNextParentLevel(const uint32_t level, const TraversalTrail& trail);
        
        void read_node(BVHNode *node, uint32_t node_ptr);
        bool isTopLevel(BVHNode *node);
        bool isLeaf(BVHNode *node);
        uint32_t calcNodePtr(uint32_t base_ptr, uint32_t idx){ return base_ptr + idx * sizeof(BVHNode); }
        void dcache_read(void* data, uint64_t addr, uint32_t size);

        Ray ray_transform(const Ray &ray, float *transform_matrix);
        float ray_tri_intersect(const Ray &ray, const Triangle &tri, float &bx, float &by, float &bz);
        float ray_box_intersect(const Ray &ray, float min_x, float min_y, float min_z, float max_x, float max_y, float max_z);

        uint32_t tlas_ptr, blas_ptr, qBvh_ptr, tri_ptr, tri_idx_ptr;

        const DCRS& dcrs_;
        RTUnit* rt_unit_;
};
}