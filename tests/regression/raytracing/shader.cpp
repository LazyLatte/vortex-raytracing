#include "shader.h"
//#include "rtx_shading.h"
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_trace.h>

void miss_shader(uint32_t rayID, kernel_arg_t *arg){
  float3_t color = arg->background_color;
  vortex::rt::setColor(rayID, color.x, color.y, color.z);
}

void closet_hit_shader(uint32_t rayID, kernel_arg_t *arg){
  auto blas_ptr = reinterpret_cast<const blas_node_t *>(arg->blas_addr);
  auto triEx_ptr = reinterpret_cast<const tri_ex_t *>(arg->triEx_addr);
  auto tex_ptr = reinterpret_cast<const uint8_t *>(arg->tex_addr);

  float3_t radiance = {0,0,0};
  float throughput = 1.0f;
  
  ray_hit_t hit;
  uint32_t d = vortex::rt::getAttr(rayID, 6);
  float dist = *reinterpret_cast<float*>(&d);
  
  uint32_t ox = vortex::rt::getAttr(rayID, 0);
  uint32_t oy = vortex::rt::getAttr(rayID, 1);
  uint32_t oz = vortex::rt::getAttr(rayID, 2);

  uint32_t dx = vortex::rt::getAttr(rayID, 3);
  uint32_t dy = vortex::rt::getAttr(rayID, 4);
  uint32_t dz = vortex::rt::getAttr(rayID, 5);

  float ro_x = *reinterpret_cast<float*>(&ox);
  float ro_y = *reinterpret_cast<float*>(&oy);
  float ro_z = *reinterpret_cast<float*>(&oz);
  float rd_x = *reinterpret_cast<float*>(&dx);
  float rd_y = *reinterpret_cast<float*>(&dy);
  float rd_z = *reinterpret_cast<float*>(&dz);

  ray_t ray;
  ray.orig = float3_t(ox, oy, oz);
  ray.dir = float3_t(dx, dy, dz);

  uint32_t bx = vortex::rt::getAttr(rayID, 7);
  uint32_t by = vortex::rt::getAttr(rayID, 8);
  uint32_t bz = vortex::rt::getAttr(rayID, 9);
  uint32_t blas_idx = vortex::rt::getAttr(rayID, 10);
  uint32_t tri_idx = vortex::rt::getAttr(rayID, 11);

  float3_t bcoords;
  bcoords.x = *reinterpret_cast<float*>(&bx);
  bcoords.y = *reinterpret_cast<float*>(&by);
  bcoords.z = *reinterpret_cast<float*>(&bz);

  //vx_printf("Dist: %f\n", dist);
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
  float3_t texColor = float3_t(1.0f); //texSample(uv, tex_pixels, blas.tex_width, blas.tex_height);
  float3_t diffuse = float3_t(1.0f); //diffuseLighting(I, N, texColor, arg->ambient_color, arg->light_color, arg->light_pos);

  auto reflectivity = blas.reflectivity;

  // add non-reflected diffuse contribution
  radiance += throughput * diffuse * (1 - reflectivity);

  // carry forward reflected energy
  throughput *= reflectivity;

  // bounce if reflective
  // if (reflectivity > 0.0f && bounce + 1 < arg->max_depth) {
  //   float3_t R = normalize(ray.dir - 2.0f * N * dot(N, ray.dir));
  //   //traceRay(I + R * 0.001f, R);
  // }

  // environment contribution for remaining throughput
  radiance += throughput * arg->background_color;

  vortex::rt::setColor(rayID, radiance.x, radiance.y, radiance.z);
}