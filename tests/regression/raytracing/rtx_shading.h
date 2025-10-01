#include "common.h"

// Sample a texture using point filtering
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

// Sample a texture using bilinear filtering
float3_t texSampleBi(const float2_t &uv, const uint32_t *pixels, uint32_t width, uint32_t height) {
  // Convert UVs to texel space
  float u = uv.x * width;
  float v = uv.y * height;

  uint32_t x0 = (uint32_t)floorf(u);
  uint32_t y0 = (uint32_t)floorf(v);
  uint32_t x1 = x0 + 1;
  uint32_t y1 = y0 + 1;

  // Compute interpolation weights
  float fu = u - x0;
  float fv = v - y0;

  // wrap coordinates
  x0 %= width;
  y0 %= height;
  x1 %= width;
  y1 %= height;

  // Sample four texels
  float3_t c00 = RGB8toRGB32F(pixels[x0 + y0 * width]);
  float3_t c10 = RGB8toRGB32F(pixels[x1 + y0 * width]);
  float3_t c01 = RGB8toRGB32F(pixels[x0 + y1 * width]);
  float3_t c11 = RGB8toRGB32F(pixels[x1 + y1 * width]);

  // Interpolate horizontally
  float3_t cx0 = c00 * (1.0f - fu) + c10 * fu;
  float3_t cx1 = c01 * (1.0f - fu) + c11 * fu;

  // Interpolate vertically
  return cx0 * (1.0f - fv) + cx1 * fv;
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