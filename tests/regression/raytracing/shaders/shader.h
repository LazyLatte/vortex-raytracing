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
  uint32_t primitiveID;
  uint32_t instanceID;

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