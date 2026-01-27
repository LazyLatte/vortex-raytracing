    
#pragma once
#include "rt_types.h"
#include "rt_trace.h"
#include "dcrs.h"

namespace vortex {
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

            uint32_t tlas_ptr, blas_ptr, qBvh_ptr, tri_ptr, tri_idx_ptr;

            const DCRS& dcrs_;
            RTUnit* rt_unit_;
    };
}