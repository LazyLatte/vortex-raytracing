#include "shader.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {

void _start(uint32_t payload_addr, kernel_arg_t *arg){
  if(payload_addr == 0x00000000) return;
  
  ray_payload_t *payload = reinterpret_cast<ray_payload_t*>(payload_addr | 0xF0000000); // address hack

  float t  = payload->t;
  float bx = payload->u;
  float by = payload->v;
  float bz = 1 - bx - by;

  auto triEx_ptr = reinterpret_cast<const tri_ex_t *>(arg->triEx_addr);
  const tri_ex_t &triEx = triEx_ptr[payload->tri_idx];

  auto blas_ptr = reinterpret_cast<const blas_node_t *>(arg->blas_addr);
  auto &blas = blas_ptr[payload->blas_idx];

  // intersection point
  float3_t I = payload->origin + payload->direction * t;

  // interpolated, transformed normal
  float3_t N = triEx.N1 * bx + triEx.N2 * by + triEx.N0 * bz;
  mat4_t invTranspose = blas.invTransform.transposed();
  N = normalize(TransformVector(N, invTranspose));

  // barycentric UV
  float2_t uv = triEx.uv1 * bx + triEx.uv2 * by + triEx.uv0 * bz;

  float3_t albedo;
  auto mat_ptr = reinterpret_cast<const material_info_t *>(arg->mat_addr);
  const material_info_t &mat = mat_ptr[triEx.texId];

  if (mat.diffuse_tex_id >= 0) {
    auto tex_ptr = reinterpret_cast<const uint8_t *>(arg->tex_addr);
    auto tex_pixels = reinterpret_cast<const uint32_t*>(tex_ptr + mat.tex_offset);
    albedo = texSample(uv, tex_pixels, mat.tex_width, mat.tex_height);
  } else {
    albedo = mat.diffuse;
  }

  float3_t ambient = albedo * arg->ambient_color;
  payload->irradiance += payload->throughput * ambient;

  // Handle Emission (Light Sources)
  if(0 /*length(mat.emissive) > 0.0f*/){
      payload->irradiance += payload->throughput * mat.emissive;
      payload->stop = true;
  } else {
      // Prepare Reflection (Only if not a light)
      payload->throughput *= albedo;
      payload->origin = I + N * 0.001f;
      payload->direction = reflect(payload->direction, N);
      payload->bounce++;

      if (payload->bounce >= arg->max_depth) {
          payload->stop = true;
      }
  }

  payload->done = true;
}

}