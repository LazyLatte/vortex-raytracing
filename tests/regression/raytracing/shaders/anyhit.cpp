#include "shader.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {
void _start(uint32_t data, kernel_arg_t *arg){
  if(data == 0) return;

  uint32_t rayID = data & 0x00FFFFFF;
  uint32_t hitID = (data & 0x0F000000) >> 24;

  vortex::rt::commit<VX_RT_ANYHIT_ACCEPT>(rayID, hitID);
}
}
