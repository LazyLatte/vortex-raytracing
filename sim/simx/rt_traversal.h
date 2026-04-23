    
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

struct BVHChildData {
    uint8_t meta;
    uint8_t qaabb[6];
};

// 64 bytes when RT_BVH_WIDTH = 6
struct BVHNode {
    uint32_t leftFirst;
#define BVH_INTERNAL   0
#define INSTANCE_LEAF  1
#define PRIMITIVE_LEAF 3
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
            uint32_t shaderIndex;   // Which intersection shader to trigger
            uint32_t primCount;     // Number of primitives in this leaf
            // uint32_t primStride;    // Size of each primitive (bytes)
            // uint32_t rayMask;       // Culling mask
            
            // "Immediate Data" cache: 49 bytes remaining
            // Can fit 1 Triangle, 2 Spheres, or 10 AABB-only pointers.
            uint8_t  payload[48];   
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
    float t = LARGE_FLOAT, u, v;
    uint32_t primitiveID; // 32bits
    uint32_t instanceID; //24 bits
};

struct ChildIntersection {
    float dist;
    uint32_t childIdx;

    ChildIntersection(float _dist, uint32_t _childIdx)
        : dist(_dist)
        , childIdx(_childIdx)
    {}
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

Ray ray_transform(const Ray &ray, float *T){
    float m00 = T[0], m01 = T[1], m02 = T[2], m03 = T[3];
    float m10 = T[4], m11 = T[5], m12 = T[6], m13 = T[7];
    float m20 = T[8], m21 = T[9], m22 = T[10], m23 = T[11];
    // float m30 = T[12], m31 = T[13], m32 = T[14], m33 = T[15];

    Ray T_ray;
    T_ray.ro_x = m00 * ray.ro_x + m01 * ray.ro_y + m02 * ray.ro_z + m03;
    T_ray.ro_y = m10 * ray.ro_x + m11 * ray.ro_y + m12 * ray.ro_z + m13;
    T_ray.ro_z = m20 * ray.ro_x + m21 * ray.ro_y + m22 * ray.ro_z + m23;

    T_ray.rd_x = m00 * ray.rd_x + m01 * ray.rd_y + m02 * ray.rd_z;
    T_ray.rd_y = m10 * ray.rd_x + m11 * ray.rd_y + m12 * ray.rd_z;
    T_ray.rd_z = m20 * ray.rd_x + m21 * ray.rd_y + m22 * ray.rd_z;
    return T_ray;
}

float ray_box_intersect(const Ray &ray, float min_x, float min_y, float min_z, float max_x, float max_y, float max_z){
    float ro_x = ray.ro_x, ro_y = ray.ro_y, ro_z = ray.ro_z;
    float rd_x = ray.rd_x, rd_y = ray.rd_y, rd_z = ray.rd_z;
    float idir_x, idir_y, idir_z, tmin, tmax, tx1, tx2, ty1, ty2, tz1, tz2;

    idir_x = 1.0f / rd_x;
    idir_y = 1.0f / rd_y;
    idir_z = 1.0f / rd_z;
    tx1 = (min_x - ro_x) * idir_x;
    tx2 = (max_x - ro_x) * idir_x;
    tmin = std::min(tx1, tx2);
    tmax = std::max(tx1, tx2);
    ty1 = (min_y - ro_y) * idir_y;
    ty2 = (max_y - ro_y) * idir_y;
    tmin = std::max(tmin, std::min(ty1, ty2));
    tmax = std::min(tmax, std::max(ty1, ty2));
    tz1 = (min_z - ro_z) * idir_z;
    tz2 = (max_z - ro_z) * idir_z;
    tmin = std::max(tmin, std::min(tz1, tz2));
    tmax = std::min(tmax, std::max(tz1, tz2));
    return tmax < tmin || tmax <= 0 ? LARGE_FLOAT : tmin;
}

void ray_nBox_intersect(const Ray &ray, BVHNode& node, float t, std::vector<ChildIntersection>& intersections){
    for(int i=0; i<RT_BVH_WIDTH; i++){
        if(node.internal.children[i].meta == 0) continue;
        float min_x = node.internal.px + std::ldexp(float(node.internal.children[i].qaabb[0]), node.ex);
        float min_y = node.internal.py + std::ldexp(float(node.internal.children[i].qaabb[1]), node.ey);
        float min_z = node.internal.pz + std::ldexp(float(node.internal.children[i].qaabb[2]), node.ez);

        float max_x = node.internal.px + std::ldexp(float(node.internal.children[i].qaabb[3]), node.ex);
        float max_y = node.internal.py + std::ldexp(float(node.internal.children[i].qaabb[4]), node.ey);
        float max_z = node.internal.pz + std::ldexp(float(node.internal.children[i].qaabb[5]), node.ez);

        float d = ray_box_intersect(ray, min_x, min_y, min_z, max_x, max_y, max_z);

        if(d < t){
            intersections.emplace_back(d, i);
        }
    }
}

float ray_tri_intersect(const Ray &ray, const Triangle &tri, float &u, float &v){
    float v0_x = tri.v0_x, v0_y = tri.v0_y, v0_z = tri.v0_z;
    float v1_x = tri.v1_x, v1_y = tri.v1_y, v1_z = tri.v1_z;
    float v2_x = tri.v2_x, v2_y = tri.v2_y, v2_z = tri.v2_z;

    float ro_x = ray.ro_x, ro_y = ray.ro_y, ro_z = ray.ro_z;
    float rd_x = ray.rd_x, rd_y = ray.rd_y, rd_z = ray.rd_z;
    
    float edge1_x = v1_x - v0_x;
    float edge1_y = v1_y - v0_y;
    float edge1_z = v1_z - v0_z;

    float edge2_x = v2_x - v0_x;
    float edge2_y = v2_y - v0_y;
    float edge2_z = v2_z - v0_z;

    float h_x = rd_y * edge2_z - rd_z * edge2_y;
    float h_y = rd_z * edge2_x - rd_x * edge2_z;
    float h_z = rd_x * edge2_y - rd_y * edge2_x;

    float a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z;
    if (fabs(a) < EPSILON){
        return LARGE_FLOAT;
    }

    float f = 1 / a;
    float s_x = ro_x - v0_x;
    float s_y = ro_y - v0_y;
    float s_z = ro_z - v0_z;

    float w1 = f * (s_x * h_x + s_y * h_y + s_z * h_z);
    if (w1 < 0 || w1 > 1){
        return LARGE_FLOAT;
    }
        
    float q_x = s_y * edge1_z - s_z * edge1_y;
    float q_y = s_z * edge1_x - s_x * edge1_z;
    float q_z = s_x * edge1_y - s_y * edge1_x;

    const float w2 = f * (rd_x * q_x + rd_y * q_y + rd_z * q_z);
    if (w2 < 0 || w1 + w2 > 1){
        return LARGE_FLOAT;
    }
        
    const float tf = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z);
    if (tf <= EPSILON){
        return LARGE_FLOAT;
    }

    u = w1;
    v = w2;
    return tf;
}

}