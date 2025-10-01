#pragma once
#include <algorithm>
#include <math.h> 
#include <queue>
#include <list>
#include <cstdint>
#include <iostream>
#define LARGE_FLOAT 1e30f
#define EPSILON 1e-6f

namespace vortex {

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
            std::list<T> stack_;
    };

    typedef struct _Hit {
        float dist = LARGE_FLOAT, bx, by, bz;
        uint32_t blasIdx, triIdx;
    }Hit;

    struct Ray {
        float ro_x, ro_y, ro_z, rd_x, rd_y, rd_z;
    };

    struct RayBuffer {
        uint32_t id;
        Ray ray;
        
        Hit hit;
        RayBuffer(){}
    };

}