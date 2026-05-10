#include "shader.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {

void _start(kernel_arg_t *arg){
  uint32_t _t = vortex::rt::get_attr<VX_RT_HIT_T>();
  uint32_t _u = vortex::rt::get_attr<VX_RT_HIT_ATTR_U>();
  uint32_t _v = vortex::rt::get_attr<VX_RT_HIT_ATTR_V>();
  uint32_t instanceID = vortex::rt::get_attr<VX_RT_HIT_INSTANCE_ID>();
  uint32_t primitiveID = vortex::rt::get_attr<VX_RT_HIT_PRIMITIVE_ID>();
  uint32_t payload_addr = vortex::rt::get_attr<VX_RT_PAYLOAD_ADDR>();

  ray_payload_t *payload = reinterpret_cast<ray_payload_t*>(payload_addr);
  float t  = *reinterpret_cast<float*>(&_t);
  float bx = *reinterpret_cast<float*>(&_u);
  float by = *reinterpret_cast<float*>(&_v);
  float bz = 1 - bx - by;

  auto triEx_ptr = reinterpret_cast<const tri_ex_t *>(arg->triEx_addr);
  const tri_ex_t &triEx = triEx_ptr[primitiveID];

  auto blas_ptr = reinterpret_cast<const blas_node_t *>(arg->blas_addr);
  auto &blas = blas_ptr[instanceID];

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

  payload->irradiance += payload->throughput * albedo;

  // Handle Emission (Light Sources)
  if(length(mat.emissive) > 0.0f){
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