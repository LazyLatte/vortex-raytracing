    
#pragma once
#include "rt_types.h"
#include "rt_trace.h"
#include <array>

#define MAX_LEVEL 32

namespace vortex {
    class RTUnit;
    class RestartTrailTraversal{
        public:
            RestartTrailTraversal(uint32_t tlas_ptr, uint32_t blas_ptr, uint32_t qBvh_ptr, uint32_t tri_ptr, uint32_t tri_idx_ptr, RTUnit* rt_unit);
            void traverse(std::pair<Ray, Hit>& ray_buf, per_thread_info &thread_info);

        private:
            void push(StackEntry e);
            bool pop();
            int32_t findNextParentLevel();
            
            void dcache_read(void* data, uint64_t addr, uint32_t size);
            void read_node(BVHNode *node);
            bool isTopLevel(BVHNode *node);
            bool isLeaf(BVHNode *node);
            uint32_t calcNodePtr(uint32_t idx){ return base_ptr + idx * sizeof(BVHNode); }
            uint32_t tlas_ptr, blas_ptr, qBvh_ptr, tri_ptr, tri_idx_ptr;

            std::array<uint32_t, MAX_LEVEL> trail; //trail[i]: 0 ~ BVH_WIDTH
            uint32_t level;
            uint32_t base_ptr;
            uint32_t node_ptr;

            ShortStack<StackEntry> traversal_stack;

            RTUnit* rt_unit_;
    };
}