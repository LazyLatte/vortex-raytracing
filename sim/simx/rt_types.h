#pragma once
#include <VX_config.h>
#include <deque>
#include <array>
#include <vector>
#include <utility>
#include <cstdint>
#include <cassert>

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
        float dist = LARGE_FLOAT, pending_dist;
        float bx, by, bz;
        uint32_t blasIdx, triIdx;
    };

    class RayList {
    public:
        RayList(){
            alive_.push_back(true); // rayID starts at 1
        }

        uint32_t allocate(){
            if (!free_.empty()) {
                uint32_t rayID = free_.back();
                free_.pop_back();
                alive_[rayID] = true;
                return rayID;
            } else {
                alive_.push_back(true);
                return alive_.size() - 1;
            }
        }

        void free(uint32_t rayID) {
            alive_[rayID] = false;
            free_.push_back(rayID);
        }

        bool valid(uint32_t rayID) const {
            return alive_[rayID];
        }

    private:
        std::vector<bool> alive_;
        std::vector<uint32_t> free_;
    };

    typedef std::array<uint32_t, MAX_TRAIL_LEVEL> TraversalTrail; //trail[i]: 0 ~ BVH_WIDTH

    class TraversalStack {
        public:
            struct Entry {
                bool last;
                uint32_t node_ptr;

                Entry(uint32_t _node_ptr, bool _last)
                    : node_ptr(_node_ptr)
                    , last(_last) 
                {}
            };

            TraversalStack(): cap_(0){}
            TraversalStack(uint32_t capacity): cap_(capacity){}

            void push(uint32_t node_ptr, bool last) {
                if (cap_ == 0) return; 
                if (stack_.size() == cap_) {
                    stack_.pop_front();  
                }
                stack_.emplace_back(node_ptr, last);
            }

            Entry pop() {
                assert(!stack_.empty());
                Entry el = std::move(stack_.back());
                stack_.pop_back();
                return el;
            }

            std::size_t size() const noexcept { return stack_.size(); }
            bool empty() const noexcept { return stack_.empty(); }
        private:
            std::size_t cap_;
            std::deque<Entry> stack_;
    };


    class ShaderQueue {
        public:
            ShaderQueue(uint32_t capacity, uint32_t width): cap_(capacity), width_(width){}

            void push(uint32_t rayID) {
                //if (cap_ == 0) return; 

                if(queue_.empty() || queue_.back().size() == width_){
                    std::vector<uint32_t> qEntry;
                    qEntry.push_back(rayID);
                    queue_.push_back(qEntry);
                }else{
                    queue_.back().push_back(rayID);
                }
            }


            std::vector<uint32_t> pop() {
                assert(!queue_.empty());
                std::vector<uint32_t> el = std::move(queue_.front());
                queue_.pop_front();
                return el;
            }

            std::size_t size() const noexcept { return queue_.size(); }
            bool empty() const noexcept { return queue_.empty(); }
        private:
            std::size_t cap_;
            std::size_t width_;
            std::deque<std::vector<uint32_t>> queue_;
    };

}