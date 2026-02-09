#include "shader.h"
#include "../rtx_shading.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>

extern "C" {

typedef void (*shader_t)(uint32_t rayID, kernel_arg_t *arg);

void _start(uint32_t rayID, kernel_arg_t *arg){
  if(rayID == 0) return;
  auto blas_ptr = reinterpret_cast<const blas_node_t *>(arg->blas_addr);
  auto triEx_ptr = reinterpret_cast<const tri_ex_t *>(arg->triEx_addr);
  auto tex_ptr = reinterpret_cast<const uint8_t *>(arg->tex_addr);
  auto sbt = reinterpret_cast<uint64_t *>(arg->sbt_addr);

  float3_t radiance = {0,0,0};
  float throughput = 1.0f;

  ray_hit_t hit;
  uint32_t d = vortex::rt::getAttr(rayID, VX_RT_HIT_DIST);
  float dist = *reinterpret_cast<float*>(&d);
  
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

  ray_t ray;
  ray.orig = float3_t(ro_x, ro_y, ro_z);
  ray.dir = float3_t(rd_x, rd_y, rd_z);

  uint32_t bx = vortex::rt::getAttr(rayID, VX_RT_HIT_BX);
  uint32_t by = vortex::rt::getAttr(rayID, VX_RT_HIT_BY);
  uint32_t bz = vortex::rt::getAttr(rayID, VX_RT_HIT_BZ);
  uint32_t blas_idx = vortex::rt::getAttr(rayID, VX_RT_HIT_BLAS_IDX);
  uint32_t tri_idx = vortex::rt::getAttr(rayID, VX_RT_HIT_TRI_IDX);

  float3_t bcoords;
  bcoords.x = *reinterpret_cast<float*>(&bx);
  bcoords.y = *reinterpret_cast<float*>(&by);
  bcoords.z = *reinterpret_cast<float*>(&bz);

  hit.dist = dist;
  hit.bcoords = bcoords;
  hit.blasIdx = blas_idx;
  hit.triIdx = tri_idx;

  // fetch instance & per-triangle data
  auto &blas = blas_ptr[hit.blasIdx];
  const tri_ex_t &triEx = triEx_ptr[hit.triIdx];

  // intersection point
  float3_t I = ray.orig + ray.dir * hit.dist;

  // interpolated, transformed normal
  float3_t N = triEx.N1 * hit.bcoords.x + triEx.N2 * hit.bcoords.y + triEx.N0 * hit.bcoords.z;
  mat4_t invTranspose = blas.invTransform.transposed();
  N = normalize(TransformVector(N, invTranspose));

  // barycentric UV
  float2_t uv = triEx.uv1 * hit.bcoords.x + triEx.uv2 * hit.bcoords.y + triEx.uv0 * hit.bcoords.z;

  // diffuse shading
  auto tex_pixels = reinterpret_cast<const uint32_t*>(tex_ptr + blas.tex_offset);
  float3_t texColor = texSample(uv, tex_pixels, blas.tex_width, blas.tex_height);
  float3_t diffuse = diffuseLighting(I, N, texColor, arg->ambient_color, arg->light_color, arg->light_pos);

  auto reflectivity = blas.reflectivity;

  // add non-reflected diffuse contribution
  radiance += throughput * diffuse * (1 - reflectivity);

  // carry forward reflected energy
  throughput *= reflectivity;

  // bounce if reflective
  uint32_t payload_addr = vortex::rt::getAttr(rayID, VX_RT_RAY_PAYLOAD_ADDR);
  ray_payload_t *payload = reinterpret_cast<ray_payload_t*>(payload_addr);
  if (reflectivity > 0.0f && payload->bounce + 1 < arg->max_depth) {
    float3_t R = normalize(ray.dir - 2.0f * N * dot(N, ray.dir));

    ray_t sec_ray;
    sec_ray.orig = I + R * 0.001f;
    sec_ray.dir = R;

    uint32_t secRayID;
    ray_payload_t secPayload;
    secPayload.bounce = payload->bounce + 1;
    vortex::rt::traceRay(sec_ray.orig.x, sec_ray.orig.y, sec_ray.orig.z, sec_ray.dir.x, sec_ray.dir.y, sec_ray.dir.z, (uint32_t)(&secPayload), secRayID);

    uint32_t ret;
    while((ret = vortex::rt::getWork()) != 0){
      uint32_t type = __builtin_ctz(ret >> 28);
      uint32_t id = ret & 0x0FFFFFFF;

      auto shader = (shader_t)(sbt[type]);
      shader(id, arg);
    }

    radiance += secPayload.color * throughput;
  }else{
    // environment contribution for remaining throughput
    radiance += arg->background_color * throughput;
  }

  payload->color = radiance;
  payload->done = true;
  vortex::rt::commit(rayID, VX_RT_COMMIT_TERM);
}

}