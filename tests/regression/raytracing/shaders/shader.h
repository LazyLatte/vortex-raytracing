#pragma once
#include <stdint.h>
#include "../common.h"

//relocatable binary!!!
typedef void (*shader_t)(uint32_t, kernel_arg_t *);

struct ray_payload_t {
  // Ray Data
  float3_t origin;
  float3_t direction;

  // Hit Data
  float t, u, v;
  uint32_t blas_idx;
  uint32_t tri_idx;

  float3_t throughput;
  float3_t irradiance;

  uint32_t bounce;
  volatile bool stop;
  volatile bool done;
};

float3_t texSample(const float2_t &uv, const uint32_t *pixels, uint32_t width, uint32_t height) {
  // Convert UVs to texel space
  uint32_t iu = uint32_t(uv.x * width);
  uint32_t iv = uint32_t(uv.y * height);

  // wrap coordinates
  iu %= width;
  iv %= height;

  // Sample texel
  uint32_t offset = (iu + iv * width);
  uint32_t texel = pixels[offset];
  return RGB8toRGB32F(texel);
}

float3_t diffuseLighting(const float3_t& pixel,
                         const float3_t& normal,
                         const float3_t& diffuse_color,
                         const float3_t& ambient_color,
                         const float3_t& light_color,
                         const float3_t& light_pos){
  float3_t L = light_pos - pixel;
  float dist = length(L);
  L *= 1.0f / dist;
  float att = 1.0f / (1.0f + dist * 0.1f);
  float NdotL = std::max(0.0f, dot(normal, L));
  return diffuse_color * (ambient_color + att * light_color * NdotL);
}

float3_t reflect(const float3_t& P, const float3_t& N){
  return normalize(P - 2.0f * N * dot(N, P));
}

float ray_tri_intersect(const ray_t &ray, const tri_t &tri, float &u, float &v){
  float v0_x = tri.v0.x, v0_y = tri.v0.y, v0_z = tri.v0.z;
  float v1_x = tri.v1.x, v1_y = tri.v1.y, v1_z = tri.v1.z;
  float v2_x = tri.v2.x, v2_y = tri.v2.y, v2_z = tri.v2.z;

  float ro_x = ray.orig.x, ro_y = ray.orig.y, ro_z = ray.orig.z;
  float rd_x = ray.dir.x,  rd_y = ray.dir.y,  rd_z = ray.dir.z;
  
  float edge1_x = v1_x - v0_x;
  float edge1_y = v1_y - v0_y;
  float edge1_z = v1_z - v0_z;

  float edge2_x = v2_x - v0_x;
  float edge2_y = v2_y - v0_y;
  float edge2_z = v2_z - v0_z;

  float h_x = rd_y * edge2_z - rd_z * edge2_y;
  float h_y = rd_z * edge2_x - rd_x * edge2_z;
  float h_z = rd_x * edge2_y - rd_y * edge2_x;

  float a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z;
  if (fabs(a) < EPSILON){
      return LARGE_FLOAT;
  }

  float f = 1 / a;
  float s_x = ro_x - v0_x;
  float s_y = ro_y - v0_y;
  float s_z = ro_z - v0_z;

  float w1 = f * (s_x * h_x + s_y * h_y + s_z * h_z);
  if (w1 < 0 || w1 > 1){
      return LARGE_FLOAT;
  }
      
  float q_x = s_y * edge1_z - s_z * edge1_y;
  float q_y = s_z * edge1_x - s_x * edge1_z;
  float q_z = s_x * edge1_y - s_y * edge1_x;

  const float w2 = f * (rd_x * q_x + rd_y * q_y + rd_z * q_z);
  if (w2 < 0 || w1 + w2 > 1){
      return LARGE_FLOAT;
  }
      
  const float tf = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z);
  if (tf <= EPSILON){
      return LARGE_FLOAT;
  }

  u = w1;
  v = w2;
  return tf;
}