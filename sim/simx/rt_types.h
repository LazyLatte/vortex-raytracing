#pragma once
#include <deque>
#include <cstdint>
#include <cassert>
namespace vortex {
    struct BVHChildData {
        uint8_t meta;
        uint8_t qaabb[6];
    };

    struct BVHNode {
        float px, py, pz;
        int8_t ex, ey, ez;
        uint8_t imask;

        uint32_t leftFirst; //First Child Idx
        uint32_t leafData; //blasIdx for TLAS, triCount for BVH
        
        BVHChildData children[4];
    };

    struct Triangle {
        float v0_x, v0_y, v0_z, v1_x, v1_y, v1_z, v2_x, v2_y, v2_z;
    };

    struct ChildIntersection {
        float dist;
        uint32_t childIdx;

        ChildIntersection(float dist, uint32_t childIdx): dist(dist), childIdx(childIdx){}
    };

    template<typename T>
    class ShortStack {
        public:
            ShortStack(uint32_t capacity): cap_(capacity){}

            void push(T val) {
                if (cap_ == 0) return; 
                if (stack_.size() == cap_) {
                    stack_.pop_front();  
                }
                stack_.push_back(std::move(val));
            }


            T pop() {
                assert(!stack_.empty());
                T el = std::move(stack_.back());
                stack_.pop_back();
                return el;
            }

            std::size_t size() const noexcept { return stack_.size(); }
            bool empty() const noexcept { return stack_.empty(); }
        private:
            std::size_t cap_;
            std::deque<T> stack_;
    };

    struct StackEntry{
        uint32_t node_ptr;
        bool last;
        StackEntry(uint32_t node_ptr, bool last): node_ptr(node_ptr), last(last) {}
    };


}