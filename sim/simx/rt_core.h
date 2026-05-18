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
            uint32_t prim_count;
            uint32_t geometryIndex;

        #define OPAQUE 0
        #define NON_OPAQUE 1
            uint8_t flags;

            uint8_t unused[47]; // Could possibly fit 1 Triangle, 2 Spheres, or 10 AABB-only pointers.
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

struct HitAttributes {
    union { float u; uint32_t attr0; };
    union { float v; uint32_t attr1; };
    uint32_t attr2, attr3, attr4, attr5, attr6, attr7;

    uint32_t& operator[](int index) {
        switch(index){
            case 0: return attr0;
            case 1: return attr1;
            case 2: return attr2;
            case 3: return attr3;
            case 4: return attr4;
            case 5: return attr5;
            case 6: return attr6;
            case 7: return attr7;
            default: return attr0;
        }
    }

    uint32_t operator[](int index) const {
        switch(index){
            case 0: return attr0;
            case 1: return attr1;
            case 2: return attr2;
            case 3: return attr3;
            case 4: return attr4;
            case 5: return attr5;
            case 6: return attr6;
            case 7: return attr7;
            default: return 0;
        }
    }
};

struct Hit {
    float t;
    HitAttributes attrs;
    uint32_t primitiveID;
    uint32_t instanceID;
    uint32_t geometryIndex;
    bool valid;

    Hit(): t(LARGE_FLOAT), primitiveID(0), instanceID(0), valid(false) {}
    Hit(float _tmax) : t(_tmax), primitiveID(0), instanceID(0), valid(false) {}
    Hit(float _t, float _u, float _v, uint32_t _primitiveID, uint32_t _instanceID) 
        : t(_t)
        , primitiveID(_primitiveID)
        , instanceID(_instanceID)
        , valid(true) 
    {
        this->attrs.u = _u;
        this->attrs.v = _v;
    }

    static Hit min(const Hit& a, const Hit& b) {
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
    uint32_t geometryIndex;
    
    uint8_t leaf_flags;
    uint32_t prim_count;
    uint32_t prim_base_id;
    uint32_t prim_batch_finished_count;

    float tmin;

    TraversalState()
        : trail({})
        , status(TraversalStatus::TRACE)
        , root_ptr(0)
        , root_level(0)
        , level(0)
        , instanceID(0)
        , tmin(0)
    {}


    TraversalState(Ray _ray, uint32_t _root_ptr, float _tmin, float _tmax)
        : ray(_ray)
        , best_hit(_tmax)
        , trail({})
        , status(TraversalStatus::TRACE)
        , root_ptr(_root_ptr)
        , root_level(0)
        , level(0)
        , instanceID(0)
        , tmin(_tmin)
    {}

    ~TraversalState(){}

    int32_t findNextParentLevel(){
        for(int32_t i = (int32_t)level - 1; i >= (int32_t)root_level; i--){
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

enum ShaderType {MISS, CLOSET, INTERSECTION, ANYHIT, NUM_OF_SHADER_TYPES};

class RTUnit;

class RTCore {
public:
    RTCore(RTUnit* rt_unit, const DCRS &dcrs);
    void allocate(uint32_t rayID, const Ray& ray, uint32_t tlas_addr, float tmin, float tmax);
    void traverse(uint32_t rayID, per_thread_info &thread_info);
    ShaderType shader_queue_pop(uint32_t out_warp[SIMD_WIDTH], uint32_t& active_lanes);
    void commit(uint32_t rayID, uint32_t hitID, Hit hit, ShaderType type, per_thread_info &thread_info);

    const Ray& get_world_ray(uint32_t rayID) const { return rays_.at(rayID); }
    const TraversalState& get_traversal_state(uint32_t rayID) const { return traversal_states_.at(rayID); }
private:
    Ray ray_transform(const Ray &ray, float *T);
    void ray_box_intersect(const Ray &ray, float min_x, float min_y, float min_z, float max_x, float max_y, float max_z, float& t_near, float& t_far);
    uint32_t ray_nBox_intersect(BVHNode& node, TraversalState& state, std::array<BoxHit, RT_BOX_INTERSECTION_WIDTH>& box_hits);
    void ray_nBox_intersect(AABB* aabbs, uint32_t primBaseID, uint32_t primCount, TraversalState& state);
    
    float ray_tri_intersect(const Ray &ray, const Triangle &tri, float &u, float &v);
    void ray_nTri_intersect(Triangle* tris, uint32_t triBaseID, uint32_t triCount, TraversalState& state);

    void shader_queue_push(ShaderType type, uint32_t rayID);
    void traverse(TraversalState& state, per_thread_info &thread_info);

    void dcache_read(void* data, uint64_t addr, uint32_t size);

    std::unordered_map<uint32_t, Ray> rays_;
    std::unordered_map<uint32_t, TraversalState> traversal_states_;
    std::array<ShaderQueue<RT_SHADER_QUEUE_CAPACITY, SIMD_WIDTH>, NUM_OF_SHADER_TYPES> shader_queues_;

    RTUnit* rt_unit_;
    const DCRS& dcrs_;
};

}