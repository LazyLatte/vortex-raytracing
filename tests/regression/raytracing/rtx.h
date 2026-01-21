#pragma once

#include "common.h"
#include "rtx_shading.h"
#include <vx_print.h>
#include <vx_spawn.h>
#include <vx_trace.h>
#include <limits>

#define BVH_STACK_SIZE 64

float3_t Trace(const ray_t &ray, const kernel_arg_t *__UNIFORM__ arg) {
  auto tri_ptr = reinterpret_cast<const tri_t *>(arg->tri_addr);
  auto bvh_ptr = reinterpret_cast<const bvh_node_t *>(arg->bvh_addr);
  auto qBvh_ptr = reinterpret_cast<const bvh_quantized_node_t *>(arg->qBvh_addr);
  auto triIdx_ptr = reinterpret_cast<const uint32_t *>(arg->triIdx_addr);
  auto tlas_ptr = reinterpret_cast<const tlas_node_t *>(arg->tlas_addr);
  auto blas_ptr = reinterpret_cast<const blas_node_t *>(arg->blas_addr);
  auto triEx_ptr = reinterpret_cast<const tri_ex_t *>(arg->triEx_addr);
  auto tex_ptr = reinterpret_cast<const uint8_t *>(arg->tex_addr);

  ray_t cur_ray = ray;

  float3_t radiance = {0,0,0};
  float throughput = 1.0f;
  
  // bounce until we hit the background or a primitive
  for (uint32_t bounce = 0; bounce < arg->max_depth; ++bounce) {
    ray_hit_t hit;
    //TLASIntersect(cur_ray, arg->tlas_root, tlas_ptr, blas_ptr, bvh_ptr, texIdx_ptr, tri_ptr, &hit);
    uint32_t ret;
    vortex::rt::traceRay(ray.orig.x, ray.orig.y, ray.orig.z, ray.dir.x, ray.dir.y, ray.dir.z, ret);
    float dist = *reinterpret_cast<float*>(&ret);

    uint32_t bx = csr_read(VX_CSR_RTX_BCOORDS1);
    uint32_t by = csr_read(VX_CSR_RTX_BCOORDS2);
    uint32_t bz = csr_read(VX_CSR_RTX_BCOORDS3);
    uint32_t blas_idx = csr_read(VX_CSR_RTX_BLAS_IDX);
    uint32_t tri_idx = csr_read(VX_CSR_RTX_TRI_IDX);

    float3_t bcoords;
    bcoords.x = *reinterpret_cast<float*>(&bx);
    bcoords.y = *reinterpret_cast<float*>(&by);
    bcoords.z = *reinterpret_cast<float*>(&bz);

    //vx_printf("Dist: %f\n", dist);
    hit.dist = dist;
    hit.bcoords = bcoords;
    hit.blasIdx = blas_idx;
    hit.triIdx = tri_idx;

    if (hit.dist == LARGE_FLOAT) {
      radiance += arg->background_color * throughput;
      break; // no hit!
    }

    // fetch instance & per-triangle data
    auto &blas = blas_ptr[hit.blasIdx];
    const tri_ex_t &triEx = triEx_ptr[hit.triIdx];

    // intersection point
    float3_t I = cur_ray.orig + cur_ray.dir * hit.dist;

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
    if (reflectivity > 0.0f && bounce + 1 < arg->max_depth) {
      float3_t R = normalize(cur_ray.dir - 2.0f * N * dot(N, cur_ray.dir));
      cur_ray.orig = I + R * 0.001f;
      cur_ray.dir = R;
      continue;
    }

    // environment contribution for remaining throughput
    radiance += throughput * arg->background_color;

    break;
  }
  
  return radiance;
}