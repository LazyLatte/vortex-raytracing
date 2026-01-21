#pragma once
#include <VX_config.h>
#include <deque>
#include <vector>
#include <utility>
#include <cstdint>
#include <cassert>

#define LARGE_FLOAT 1e30f

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


    struct Hit {
        float dist = LARGE_FLOAT, bx, by, bz;
        uint32_t blasIdx, triIdx;
    };

    struct Ray {
        float ro_x, ro_y, ro_z, rd_x, rd_y, rd_z;
    };

    class RayBuffer {
    public:
        uint32_t allocate(){
            if (!free_.empty()) {
                uint32_t rayID = free_.back();
                free_.pop_back();
                alive_[rayID] = true;
                return rayID;
            } else {
                data_.push_back(std::make_pair(Ray{}, Hit{}));
                alive_.push_back(true);
                return data_.size() - 1;
            }
        }

        void set_ray_x(uint32_t rayID, float ro_x, float rd_x){
            data_[rayID].first.ro_x = ro_x;
            data_[rayID].first.rd_x = rd_x;
        }

        void set_ray_y(uint32_t rayID, float ro_y, float rd_y){
            data_[rayID].first.ro_y = ro_y;
            data_[rayID].first.rd_y = rd_y;
        }

        void set_ray_z(uint32_t rayID, float ro_z, float rd_z){
            data_[rayID].first.ro_z = ro_z;
            data_[rayID].first.rd_z = rd_z;
        }

        void remove(uint32_t rayID) {
            alive_[rayID] = false;
            free_.push_back(rayID);
        }

        std::pair<Ray, Hit>& get(uint32_t rayID) {
            return data_[rayID];
        }

        bool valid(uint32_t rayID) const {
            return alive_[rayID];
        }

    private:
        std::vector<std::pair<Ray, Hit>> data_;
        std::vector<bool> alive_;
        std::vector<uint32_t> free_;
    };

}