#include "sphere.h"
#include "cornellbox.h"
#include "bunny.h"
#include "sponza.h"
#include "chestnut.h"
#include "ship.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {

void _start(kernel_arg_t *arg){
    uint32_t payload_addr = vortex::rt::get_attr<VX_RT_PAYLOAD_ADDR>();

    ray_payload_t *payload = reinterpret_cast<ray_payload_t*>(payload_addr);

    uint32_t geometry_idx = vortex::rt::get_attr<VX_RT_HIT_GEOMETRY_INDEX>();
    uint32_t shader_record_stride = payload->shader_record_stride;
    uint32_t shader_record_offset = payload->shader_record_offset;
    uint32_t shader_idx = geometry_idx;// * shader_record_stride + shader_record_offset;

    switch(shader_idx){
        case Geomrtry::Sphere: // Sphere
            Shader::Sphere::CHS(payload, arg);
            break;
        case Geomrtry::CornellBox: // CornellBox
            Shader::CornellBox::CHS(payload, arg);
            break;
        case Geomrtry::Bunny: // Bunny
            Shader::Bunny::CHS(payload, arg);
            break;
        case Geomrtry::Sponza: // Sponza
            Shader::Sponza::CHS(payload, arg);
            break;
        case Geomrtry::Chestnut: // Chestnut
            Shader::Chestnut::CHS(payload, arg);
            break;
        case Geomrtry::Ship: // Ship
            Shader::Ship::CHS(payload, arg);
            break;
        default: break;
    }

    payload->done = true;
}

}