#include "shader.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {
void _start(uint32_t rayID, kernel_arg_t *arg){
    if(rayID == 0) return;
    uint32_t ox = vortex::rt::getAttr(rayID, VX_RT_RAY_RO_X);
    uint32_t oy = vortex::rt::getAttr(rayID, VX_RT_RAY_RO_Y);
    uint32_t oz = vortex::rt::getAttr(rayID, VX_RT_RAY_RO_Z);

    uint32_t dx = vortex::rt::getAttr(rayID, VX_RT_RAY_RD_X);
    uint32_t dy = vortex::rt::getAttr(rayID, VX_RT_RAY_RD_Y);
    uint32_t dz = vortex::rt::getAttr(rayID, VX_RT_RAY_RD_Z);

    float ro_x = *reinterpret_cast<float*>(&ox);
    float ro_y = *reinterpret_cast<float*>(&oy);
    float ro_z = *reinterpret_cast<float*>(&oz);
    float rd_x = *reinterpret_cast<float*>(&dx);
    float rd_y = *reinterpret_cast<float*>(&dy);
    float rd_z = *reinterpret_cast<float*>(&dz);

    ray_t ray;
    ray.orig = make_float3(ro_x, ro_y, ro_z);
    ray.dir = make_float3(rd_x, rd_y, rd_z);
    
    uint32_t node_addr = vortex::rt::getAttr(rayID, VX_RT_HIT_TRI_IDX);
    bvh_quantized_node_t* node = reinterpret_cast<bvh_quantized_node_t*>(node_addr);

    uint32_t bt = vortex::rt::getAttr(rayID, VX_RT_HIT_T_BEST);

    hit_t hit;
    hit.t = *reinterpret_cast<float*>(&bt);

    // intersection logic
    auto tri_ptr = reinterpret_cast<const tri_t *>(arg->tri_addr);

    uint32_t leftFirst = node->leftFirst;
    uint32_t triCount = node->leaf.primCount;

    bool hit_found = false;
    for (uint32_t i = 0; i < triCount; ++i) {
        uint32_t triIdx = leftFirst + i;                    
        const tri_t& tri = tri_ptr[triIdx];
        
        float v0_x = tri.v0.x, v0_y = tri.v0.y, v0_z = tri.v0.z;
        float v1_x = tri.v1.x, v1_y = tri.v1.y, v1_z = tri.v1.z;
        float v2_x = tri.v2.x, v2_y = tri.v2.y, v2_z = tri.v2.z;

        float ro_x = ray.orig.x, ro_y = ray.orig.y, ro_z = ray.orig.z;
        float rd_x = ray.dir.x,  rd_y = ray.dir.y,  rd_z = ray.dir.z;
        
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

        float f = 1 / a;
        float s_x = ro_x - v0_x;
        float s_y = ro_y - v0_y;
        float s_z = ro_z - v0_z;

        float w1 = f * (s_x * h_x + s_y * h_y + s_z * h_z);
            
        float q_x = s_y * edge1_z - s_z * edge1_y;
        float q_y = s_z * edge1_x - s_x * edge1_z;
        float q_z = s_x * edge1_y - s_y * edge1_x;

        const float w2 = f * (rd_x * q_x + rd_y * q_y + rd_z * q_z); 
        const float tf = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z);

        float u = w1;
        float v = w2;
        float t = (fabs(a) < EPSILON || w1 < 0 || w1 > 1 || w2 < 0 || w1 + w2 > 1 || tf <= EPSILON) ?  LARGE_FLOAT : tf;

        if (t < hit.t) {
            hit.t = t;
            hit.u = u;
            hit.v = v;
            hit.primitiveID = triIdx;

            hit_found = true;
        }
    }

    if(hit_found){
        vortex::rt::commit(rayID, (uint32_t)(&hit));
    }else{
        vortex::rt::commit(rayID, VX_RT_INTERSECTION_IGNORE);
    }
}
}
