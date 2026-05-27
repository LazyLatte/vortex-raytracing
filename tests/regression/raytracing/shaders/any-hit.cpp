#include "chestnut.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {
void _start(kernel_arg_t *arg){
    uint32_t payload_addr = vortex::rt::get_attr<VX_RT_PAYLOAD_ADDR>();

    ray_payload_t *payload = reinterpret_cast<ray_payload_t*>(payload_addr);

    uint32_t geometry_idx = vortex::rt::get_attr<VX_RT_HIT_GEOMETRY_INDEX>();

    switch(geometry_idx){
        case Geomrtry::Chestnut:
            Shader::Chestnut::AHS(payload, arg);
            break;
        default: break;
    }
}
}
