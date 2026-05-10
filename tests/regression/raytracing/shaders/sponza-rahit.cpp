#include "shader.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {
void _start(kernel_arg_t *arg){
  vortex::rt::commit<VX_RT_ANYHIT_ACCEPT>();
}
}
