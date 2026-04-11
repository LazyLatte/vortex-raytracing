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
    
    float best_t = vortex::rt::getAttr(rayID, VX_RT_BEST_HIT_T);

    // uint32_t leftFirst = vortex::rt::getAttr(rayID, VX_RT_LEFT_FIRST);
    // uint32_t triCount  = vortex::rt::getAttr(rayID, VX_RT_TRI_COUNT);
    // uint32_t blasIdx   = vortex::rt::getAttr(rayID, VX_RT_BLAS_IDX);

    bool hit_found = false;
    float hit_t, hit_u, hit_v;
    uint32_t hit_triIdx;

    // 3. The Ported Loop
    auto tri_ptr = reinterpret_cast<const tri_t *>(arg->tri_addr);

    for (uint32_t i = 0; i < triCount; ++i) {
        uint32_t triIdx = leftFirst + i;
        const tri_t& tri = tri_ptr[triIdx];

        float u, v;
        float t = ray_tri_intersect(ray, tri, u, v);
        
        // Only consider hits closer than the current best known to hardware
        if (t > 0.0f && t < best_t) {
            best_t = t;
            hit_t = t;
            hit_u = u;
            hit_v = v;
            hit_triIdx = triIdx;
            hit_found = true;
        }
    }

    // 4. Report back to Hardware
    if (hit_found) {
        // This is the equivalent of 'reportIntersection' in Vulkan
        vortex::rt::reportHit(rayID, hit_t, hit_u, hit_v, hit_triIdx, blasIdx);
    } else {
        vortex::rt::commit(rayID, VX_RT_INTERSECTION_IGNORE);
    }
    
    // Crucial: Tell producer this 'sub-task' is done 
    // (Note: For intersection, this often triggers the RT unit to resume traversal)
}
}
