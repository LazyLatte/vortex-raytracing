#include "shader.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {
void _start(uint32_t rayID, kernel_arg_t *arg){
  if(rayID == 0) return;
  vortex::rt::commit<VX_RT_ANYHIT_ACCEPT>(rayID);
}
}
