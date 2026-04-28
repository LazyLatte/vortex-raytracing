#include "shader.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {
  void _start(uint32_t rayID, kernel_arg_t *arg){
    if(rayID == 0) return;

    uint32_t payload_addr = vortex::rt::get_attr<VX_RT_PAYLOAD_ADDR>(rayID);
    vortex::rt::release_ray(rayID);
    ray_payload_t *payload = reinterpret_cast<ray_payload_t*>(payload_addr);

    payload->irradiance += payload->throughput * arg->background_color;
    payload->stop = true;
    payload->done = true;
  }
}

