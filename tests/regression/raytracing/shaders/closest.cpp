#include "shader.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {

void _start(uint32_t rayID, kernel_arg_t *arg){
  if(rayID == 0) return;
  auto blas_ptr = reinterpret_cast<const blas_node_t *>(arg->blas_addr);
  auto triEx_ptr = reinterpret_cast<const tri_ex_t *>(arg->triEx_addr);
  auto mat_ptr = reinterpret_cast<const material_info_t *>(arg->mat_addr);
  auto tex_ptr = reinterpret_cast<const uint8_t *>(arg->tex_addr);

  uint32_t payload_addr = vortex::rt::getAttr(rayID, VX_RT_RAY_PAYLOAD_ADDR);
  ray_payload_t *payload = reinterpret_cast<ray_payload_t*>(payload_addr);

  uint32_t ox = vortex::rt::getAttr(rayID, VX_RT_RAY_RO_X);
  uint32_t oy = vortex::rt::getAttr(rayID, VX_RT_RAY_RO_Y);
  uint32_t oz = vortex::rt::getAttr(rayID, VX_RT_RAY_RO_Z);

  uint32_t dx = vortex::rt::getAttr(rayID, VX_RT_RAY_RD_X);
  uint32_t dy = vortex::rt::getAttr(rayID, VX_RT_RAY_RD_Y);
  uint32_t dz = vortex::rt::getAttr(rayID, VX_RT_RAY_RD_Z);

  float ro_x = *reinterpret_cast<float*>(&ox);
  float ro_y = *reinterpret_cast<float*>(&oy);
  float ro_z = *reinterpret_cast<float*>(&oz);
  float rd_x = *reinterpret_cast<float*>(&dx);
  float rd_y = *reinterpret_cast<float*>(&dy);
  float rd_z = *reinterpret_cast<float*>(&dz);

  float3_t ray_orig = make_float3(ro_x, ro_y, ro_z);
  float3_t ray_dir = make_float3(rd_x, rd_y, rd_z);

  float t  = payload->t;
  float bx = payload->u;
  float by = payload->v;
  float bz = 1 - bx - by;

  uint32_t blas_idx = payload->blas_idx;
  uint32_t tri_idx = payload->tri_idx;

  // fetch instance & per-triangle data
  auto &blas = blas_ptr[blas_idx];
  const tri_ex_t &triEx = triEx_ptr[tri_idx];
  const material_info_t &mat = mat_ptr[triEx.texId];

  // intersection point
  float3_t I = ray_orig + ray_dir * t;

  // interpolated, transformed normal
  float3_t N = triEx.N1 * bx + triEx.N2 * by + triEx.N0 * bz;
  mat4_t invTranspose = blas.invTransform.transposed();
  N = normalize(TransformVector(N, invTranspose));

  // barycentric UV
  float2_t uv = triEx.uv1 * bx + triEx.uv2 * by + triEx.uv0 * bz;

  float3_t texColor;
  if (mat.diffuse_tex_id >= 0) {
      auto tex_pixels = reinterpret_cast<const uint32_t*>(tex_ptr + mat.tex_offset);
      texColor = texSample(uv, tex_pixels, mat.tex_width, mat.tex_height);
  } else {
      texColor = mat.diffuse;
  }

  // diffuse shading
  float3_t diffuse = diffuseLighting(I, N, texColor, arg->ambient_color, arg->light_color, arg->light_pos);

  float reflectivity = blas.reflectivity;
  float throughput = 1.0f;
  // add non-reflected diffuse contribution
  float3_t radiance = throughput * diffuse * (1.0 - reflectivity);

  // carry forward reflected energy
  throughput *= reflectivity;

  // bounce if reflective
  if (reflectivity > 0.0f && payload->bounce + 1 < arg->max_depth) {
    float3_t R = normalize(ray_dir - 2.0f * N * dot(N, ray_dir));

    float3_t sec_ray_orig = I + R * 0.001f;
    float3_t sec_ray_dir = R;

    ray_payload_t sec_payload;
    sec_payload.done = false;
    sec_payload.bounce = payload->bounce + 1;
    
    uint32_t sec_rayID;
    vortex::rt::traceRay(
        sec_ray_orig.x, sec_ray_orig.y, sec_ray_orig.z, 
        sec_ray_dir.x, sec_ray_dir.y, sec_ray_dir.z, 
        (uint32_t)(&sec_payload), 
        sec_rayID
    );

    while(!vx_vote_all(sec_payload.done));
    radiance += sec_payload.color * throughput;
  }else{
    // environment contribution for remaining throughput
    radiance += arg->background_color * throughput;
  }

  payload->color = radiance;
  payload->done = true;
  vortex::rt::commit(rayID, VX_RT_COMMIT_TERM);
}

}