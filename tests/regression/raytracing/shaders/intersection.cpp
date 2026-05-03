#include "shader.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {
void _start(kernel_arg_t *arg){
    uint32_t ox = vortex::rt::get_attr<VX_RT_OBJECT_RAY_RO_X>();
    uint32_t oy = vortex::rt::get_attr<VX_RT_OBJECT_RAY_RO_Y>();
    uint32_t oz = vortex::rt::get_attr<VX_RT_OBJECT_RAY_RO_Z>();
    uint32_t dx = vortex::rt::get_attr<VX_RT_OBJECT_RAY_RD_X>();
    uint32_t dy = vortex::rt::get_attr<VX_RT_OBJECT_RAY_RD_Y>();
    uint32_t dz = vortex::rt::get_attr<VX_RT_OBJECT_RAY_RD_Z>();
    uint32_t _tmin = vortex::rt::get_attr<VX_RT_T_MIN>();
    uint32_t _tmax = vortex::rt::get_attr<VX_RT_T_MAX>();
    uint32_t primitiveID = vortex::rt::get_attr<VX_RT_HIT_PRIMITIVE_ID>();

    float ro_x = *reinterpret_cast<float*>(&ox);
    float ro_y = *reinterpret_cast<float*>(&oy);
    float ro_z = *reinterpret_cast<float*>(&oz);
    float rd_x = *reinterpret_cast<float*>(&dx);
    float rd_y = *reinterpret_cast<float*>(&dy);
    float rd_z = *reinterpret_cast<float*>(&dz);

    float tmin = *reinterpret_cast<float*>(&_tmin);
    float tmax = *reinterpret_cast<float*>(&_tmax);

    auto tri_ptr = reinterpret_cast<const tri_t *>(arg->tri_addr);
    const tri_t& tri = tri_ptr[primitiveID];

    // intersection logic
    float v0_x = tri.v0.x, v0_y = tri.v0.y, v0_z = tri.v0.z;
    float v1_x = tri.v1.x, v1_y = tri.v1.y, v1_z = tri.v1.z;
    float v2_x = tri.v2.x, v2_y = tri.v2.y, v2_z = tri.v2.z;
    
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
    float t = (fabs(a) < EPSILON || w1 < 0 || w1 > 1 || w2 < 0 || w1 + w2 > 1 || tf <= EPSILON) ?  tmax : tf;
    

    if (t > tmin && t < tmax) {
        vortex::rt::set_attr<VX_RT_HIT_ATTRIBUTES>(t, u, v);
        vortex::rt::commit<VX_RT_INTERSECTION_ACCEPT>();
    }else{
        vortex::rt::commit<VX_RT_INTERSECTION_IGNORE>();
    }
}
}
