#pragma once
#include "rt_trace.h"
#include "dcrs.h"
#include "types.h"
#include <array>
#include <cmath> 
#include <algorithm>

#define EPSILON 1e-6f
#define LARGE_FLOAT 1e30f
#define MAX_TRAIL_LEVEL 32

namespace vortex {

struct AABB { float min[3], max[3]; };

struct BVHChildData {
    uint8_t meta;
    uint8_t qaabb[6];
};

// 64 bytes when RT_BVH_WIDTH = 6
struct BVHNode {
    uint32_t leftFirst;
#define BVH_INTERNAL    0
#define INSTANCE_LEAF   1
#define TRIANGLE_LEAF   3
#define PROCEDURAL_LEAF 4
    uint8_t type;
    int8_t ex, ey, ez;

    union {
        // --- INTERNAL NODE ---
        struct {
            float px, py, pz;
            BVHChildData children[RT_BVH_WIDTH];
            uint8_t padding[2];
        } internal;

        // --- LEAF ---
        struct {
            union {
                uint32_t instanceID;
                uint32_t prim_count;
            };

            uint32_t shader_idx;

        #define OPAQUE 0
        #define NON_OPAQUE 1
            uint8_t flags;
            // uint32_t primStride;    // Size of each primitive (bytes)
            // uint32_t rayMask;       // Culling mask
            
            // Can fit 1 Triangle, 2 Spheres, or 10 AABB-only pointers.
            uint8_t payload[47];   
        } leaf;
    };
};

struct BLASNode {
    uint32_t bvh_offset;
    float invTransform[12];

    float transform[12]; // unused
    uint32_t mat_offset; // unused
    uint8_t padding[24]; // unused
};

struct Triangle {
    float v0_x, v0_y, v0_z, v1_x, v1_y, v1_z, v2_x, v2_y, v2_z;
};

struct Ray {
    float ro_x, ro_y, ro_z, rd_x, rd_y, rd_z;
};

struct Hit {
    float t, u, v;
    uint32_t primitiveID; // 32bits
    uint32_t instanceID; // 24bits
    bool valid;

    Hit()
        : t(LARGE_FLOAT)
        , u(0.0)
        , v(0.0)
        , primitiveID(0)
        , instanceID(0)
        , valid(false) 
    {}

    Hit(float _tmax) 
        : t(_tmax)
        , u(0.0)
        , v(0.0)
        , primitiveID(0)
        , instanceID(0)
        , valid(false) 
    {}

    // Hit(uint32_t _primitiveID, uint32_t _instanceID) 
    //     , primitiveID(_primitiveID)
    //     , instanceID(_instanceID)
    //     , valid(true) 
    // {}

    Hit(float _t, float _u, float _v, uint32_t _primitiveID, uint32_t _instanceID) 
        : t(_t)
        , u(_u)
        , v(_v)
        , primitiveID(_primitiveID)
        , instanceID(_instanceID)
        , valid(true) 
    {}

    static Hit compare(const Hit& a, const Hit& b) {
        if(!a.valid) return b;
        if(!b.valid) return a; 
        return (a.t < b.t) ? a : b; 
    }
};

struct BoxHit {
    float t;
    uint32_t idx; // child index: 0 ~ RT_BVH_WIDTH-1
    bool valid;

    BoxHit(): t(LARGE_FLOAT), idx(0), valid(false) {}
    BoxHit(float _t, uint32_t _idx) : t(_t), idx(_idx), valid(true) {}

    static bool compare(const BoxHit& a, const BoxHit& b) { 
        if (a.valid && b.valid) return a.t < b.t;
        return a.valid > b.valid;
    }
};

typedef ShortStack<uint32_t, RT_STACK_SIZE> TraversalStack;
typedef std::array<uint32_t, MAX_TRAIL_LEVEL> TraversalTrail; //trail[i]: 0 ~ BVH_WIDTH

enum TraversalStatus { TRACE, FINISHED, RESTART, INSTANCE_HIT, TRI_LEAF_HIT, PROCEDURAL_LEAF_HIT };

struct TraversalState {
    Ray ray;
    Hit best_hit;
    Hit prim_hit[RT_BOX_INTERSECTION_WIDTH];
    TraversalTrail trail;
    TraversalStack stack;
    TraversalStatus status;
    uint32_t root_ptr;
    uint32_t root_level;
    uint32_t level;
    uint32_t instanceID;

    uint8_t leaf_flags;
    uint32_t prim_count;
    uint32_t prim_base_id;
    uint32_t prim_batch_finished_count;

    float tmin;

    TraversalState()
        : trail({})
        , root_ptr(0)
        , root_level(0)
        , level(0)
        , instanceID(0)
        , tmin(0)
        , status(TraversalStatus::TRACE)
    {}


    TraversalState(Ray _ray, uint32_t _root_ptr, float _tmin, float _tmax)
        : ray(_ray)
        , trail({})
        , root_ptr(_root_ptr)
        , root_level(0)
        , level(0)
        , instanceID(0)
        , tmin(_tmin)
        , best_hit(_tmax)
        , status(TraversalStatus::TRACE)
    {}

    ~TraversalState(){}

    int32_t findNextParentLevel(){
        for(int i=level-1; i>=root_level; i--){
            if(trail[i] != RT_BVH_WIDTH){
                return i;
            }
        }
        return -1;
    }

    uint32_t pop(){
        int32_t parentLevel = findNextParentLevel();

        if(parentLevel < 0){
            status = TraversalStatus::FINISHED;
            return 0;
        }

        trail[parentLevel]++;

        for(int i=parentLevel+1; i<MAX_TRAIL_LEVEL; i++){
            trail[i] = 0;
        }

        if(stack.empty()){
            status = TraversalStatus::RESTART;
            return 0;
        }

        uint32_t e = stack.pop();
        uint32_t node_ptr = e & 0xFFFFFFFE;

        if(e & 1){
            trail[parentLevel] = RT_BVH_WIDTH;
        }

        level = parentLevel + 1;
        status = TraversalStatus::TRACE;
        
        return node_ptr;
    }

    bool has_prim_hit(){
        bool h = false;
        for(int i=0; i<RT_BOX_INTERSECTION_WIDTH; i++){
            h |= prim_hit[i].valid;
        }
        return h;
    }
};

enum ShaderType {MISS, CLOSET, INTERSECTION, ANYHIT, ShaderTypes};

class RTUnit;

class RTCore {
public:
    RTCore(RTUnit* rt_unit, const DCRS &dcrs);
    uint32_t traverse(uint32_t rayID, per_thread_info &thread_info);
    uint32_t commit(uint32_t rayID, uint32_t hitID, Hit hit, ShaderType type, per_thread_info &thread_info);

    std::unordered_map<uint32_t, Ray> rays_;
    std::unordered_map<uint32_t, TraversalState> traversal_states_; // Stored in RT Core Latches/Registers

private:
    Ray ray_transform(const Ray &ray, float *T);
    void ray_box_intersect(const Ray &ray, float min_x, float min_y, float min_z, float max_x, float max_y, float max_z, float& t_near, float& t_far);
    uint32_t ray_nBox_intersect(BVHNode& node, TraversalState& state, std::array<BoxHit, RT_BOX_INTERSECTION_WIDTH>& box_hits);
    void ray_nBox_intersect(AABB* aabbs, uint32_t primBaseID, uint32_t primCount, TraversalState& state);
    
    float ray_tri_intersect(const Ray &ray, const Triangle &tri, float &u, float &v);
    void ray_nTri_intersect(Triangle* tris, uint32_t triBaseID, uint32_t triCount, TraversalState& state);

    void traverse(TraversalState& state, per_thread_info &thread_info);

    void dcache_read(void* data, uint64_t addr, uint32_t size);

    bool isLeaf(BVHNode *node){ return node->type != BVH_INTERNAL; }
    bool isInstanceLeaf(BVHNode *node){ return node->type == INSTANCE_LEAF; }
    bool isProceduralLeaf(BVHNode *node){ return node->type == PROCEDURAL_LEAF; }

    uint32_t tlas_ptr, blas_ptr, bvh_ptr, tri_ptr, aabb_ptr;

    const DCRS& dcrs_;
    RTUnit* rt_unit_;
};

}