#pragma once
#include <stdint.h>
#include "../common.h"

//relocatable binary!!!
typedef void (*shader_t)(kernel_arg_t *);

struct ray_payload_t {
  // Ray Data
  float3_t origin;
  float3_t direction;

  uint32_t shader_miss_idx;
  uint32_t shader_record_stride;
  uint32_t shader_record_offset;

  float3_t throughput;
  float3_t irradiance;

  uint32_t bounce;
  volatile bool stop;
  volatile bool done;
};

inline float3_t texSample(const float2_t &uv, const uint32_t *pixels, uint32_t width, uint32_t height) {
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

inline float3_t diffuseLighting(const float3_t& pixel,
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

inline float3_t reflect(const float3_t& P, const float3_t& N){ return normalize(P - 2.0f * N * dot(N, P)); }

inline float3_t calcNormal(const tri_t& tri){
  float3_t edge1 = tri.v1 - tri.v0;
  float3_t edge2 = tri.v2 - tri.v0;

  return normalize(cross(edge1, edge2));
}

// Cosine-weighted hemisphere direction around N.
// Uses only sqrtf — no sinf/cosf — for GPU kernel compatibility.
// seed is advanced in place; seed must be non-zero (caller ensures seed |= 1u).
// Monte Carlo weight for Lambertian BRDF (f = albedo/pi) simplifies to albedo.
inline float3_t cosineSampleHemisphere(const float3_t& N, uint32_t& seed) {
  seed |= 1u;  // XOR32 loops forever at 0 — guarantee non-zero
  float x, y, z, sq = 2.0f;
  for (int i = 0; i < 32 && (sq > 1.0f || sq < 1e-8f); ++i) {
    x = RandomFloat(&seed) * 2.0f - 1.0f;
    y = RandomFloat(&seed) * 2.0f - 1.0f;
    z = RandomFloat(&seed) * 2.0f - 1.0f;
    sq = x*x + y*y + z*z;
  }
  if (sq < 1e-8f || sq > 1.0f) return N;  // fallback — probability < 10^-10
  float3_t v = float3_t(x, y, z) * (1.0f / sqrtf(sq));
  if (dot(v, N) < 0.0f) v = -v;
  return normalize(N + v);
}