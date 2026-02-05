#include "shader.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {
  void _start(uint32_t rayID, kernel_arg_t *arg){
    if(rayID == 0) return;
    uint32_t payload_addr = vortex::rt::getAttr(rayID, VX_RT_RAY_PAYLOAD_ADDR);
    ray_payload_t *payload = reinterpret_cast<ray_payload_t*>(payload_addr);

    payload->color = arg->background_color;
    vortex::rt::commit(rayID, VX_RT_COMMIT_TERM);
  }
}

