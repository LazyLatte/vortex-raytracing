#include "shader.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {
  void _start(uint32_t payload_addr, kernel_arg_t *arg){
    if(payload_addr == 0) return;
    ray_payload_t *payload = reinterpret_cast<ray_payload_t*>(payload_addr | 0xF0000000); // address hack

    payload->irradiance += payload->throughput * arg->background_color;
    payload->stop = true;
    payload->done = true;
  }
}

