#include "shader.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {
void _start(uint32_t rayID, kernel_arg_t *arg){
  if(rayID == 0) return;
  auto triEx_ptr = reinterpret_cast<const tri_ex_t *>(arg->triEx_addr);
  auto tex_ptr = reinterpret_cast<const uint8_t *>(arg->tex_addr);

  uint32_t bx = vortex::rt::getAttr(rayID, VX_RT_HIT_BX);
  uint32_t by = vortex::rt::getAttr(rayID, VX_RT_HIT_BY);
  uint32_t bz = vortex::rt::getAttr(rayID, VX_RT_HIT_BZ);
  uint32_t blas_idx = vortex::rt::getAttr(rayID, VX_RT_HIT_BLAS_IDX);
  uint32_t tri_idx = vortex::rt::getAttr(rayID, VX_RT_HIT_TRI_IDX);

  float3_t bcoords;
  bcoords.x = *reinterpret_cast<float*>(&bx);
  bcoords.y = *reinterpret_cast<float*>(&by);
  bcoords.z = *reinterpret_cast<float*>(&bz);

  const tri_ex_t &triEx = triEx_ptr[tri_idx];
  // 1. Get the UV coordinates of the current hit point
  float2_t uv = triEx.uv1 * bcoords.x + triEx.uv2 * bcoords.y + triEx.uv0 * bcoords.z;

  // 2. Sample the alpha texture (0.0 = transparent, 1.0 = opaque)
  float alpha = 1.0f;

  float alpha_threshold = 0.5f;
  if (alpha < alpha_threshold) {
    vortex::rt::commit(rayID, VX_RT_COMMIT_CONT);
  }else{
    vortex::rt::commit(rayID, VX_RT_COMMIT_ACCEPT);
  }
  
}
}
