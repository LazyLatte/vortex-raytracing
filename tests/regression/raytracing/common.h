#pragma once

#include "geometry.h"
#include <stdint.h>
#include <cmath>

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    return _ret;                                             \
  } while (false)

#define INVALID_IDX 0xFFFFFFFF

#define BVH_WIDTH 6
#define MAX_LEAF_PRIMITIVES 1

enum MaterialModel : int {
  MaterialLambertian  = 0,
  MaterialMetallic    = 1,
  MaterialDielectric  = 2,
};

enum Geomrtry : uint32_t {
  Sphere = 0,
  CornellBox = 1,
  Bunny = 2,
  Sponza = 3,
  Chestnut = 4,
  Ship = 5
};

//enum ShapeType : uint32_t { Sphere = 0 };

struct shape_t {
  float3_t center;
  float    radius;
  //uint32_t type;   // ShapeType
};

struct material_info_t {
  float3_t ambient;
  float3_t diffuse;
  float3_t specular;
  float3_t emissive;
  
  float shininess;
  float ior;
  float dissolve; 
  float reflectivity;

  int diffuse_tex_id;
  int illum;
  uint32_t tex_width;
  uint32_t tex_height;
  uint64_t tex_offset;
  uint32_t has_alpha;
};

// additional triangle data, for texturing and shading
struct tri_ex_t {
  float3_t N0, N1, N2;
  float2_t uv0, uv1, uv2;
  uint32_t texId;
};

struct child_data_t {
  uint8_t meta;
  uint8_t qaabb[6];
};

// Compressed wide BVH node struct
struct cwbvh_node_t {
  uint32_t leftFirst;

#define BVH_INTERNAL    0
#define INSTANCE_LEAF   1
#define TRIANGLE_LEAF   3
#define PROCEDURAL_LEAF 4
  uint8_t type;
  int8_t ex, ey, ez;

  union {
      // --- INTERNAL NODE ---
      struct {
          float3_t origin;
          child_data_t children[BVH_WIDTH];
          uint8_t padding[2];
      } internal;

      // --- LEAF ---
      struct {
          uint32_t primCount;
          uint32_t geometryIndex;

        #define OPAQUE 0
        #define NON_OPAQUE 1
          uint8_t flags;
          uint8_t unused[47];   
      } leaf;
  };
};

// BVH node struct
struct bvh_node_t {
  float3_t aabbMin, aabbMax;
  float3_t centroidMin, centroidMax;

  uint32_t leftFirst;
  uint32_t triCount;
  uint32_t childCount = 0;

  bool isLeaf() const { return triCount != 0; }

  float calculateNodeCost() const {
    return surfaceArea(aabbMin, aabbMax) * triCount;
  }
};

// bottom-level acceleration structure
struct blas_node_t {
  uint32_t bvh_offset = 0;
  mat3x4_t invTransform;
  mat3x4_t transform;
  uint32_t mat_offset = 0;

  uint8_t padding[24];

  void applyTransform(const mat4_t &T) {
    mat4_t m = transform.toMat4();
    m = T * m;
    this->transform = m;
    this->invTransform = m.inverted();
  }
};

// top-level acceleration structure
struct tlas_node_t {
  float3_t aabbMin;
  uint32_t leftFirst;

  float3_t aabbMax;
  uint32_t triCount;
  uint32_t start, end;
  uint32_t childCount;
  bool isLeaf() const { return childCount == 0; }

  float calculateNodeCost() const {
    return surfaceArea(aabbMin, aabbMax) * triCount;
  }
};

inline uint32_t WangHash(uint32_t s) {
  s = (s ^ 61) ^ (s >> 16);
  s *= 9, s = s ^ (s >> 4);
  s *= 0x27d4eb2d;
  s = s ^ (s >> 15);
  return s;
}

inline uint32_t RandomInt(uint32_t *s) {
  // Marsaglia's XOR32 RNG
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}

inline float RandomFloat(uint32_t *s) {
  return RandomInt(s) * 2.3283064365387e-10f; // = 1 / (2^32-1)
}

// Per-channel Reinhard: maps [0, ∞) → [0, 1) without hard clipping.
inline float3_t reinhardTonemap(float3_t c) {
  return float3_t(c.x / (1.0f + c.x),
                  c.y / (1.0f + c.y),
                  c.z / (1.0f + c.z));
}

// Gamma correction using sqrtf (≈ γ 2.0, GPU-safe — avoids powf).
inline float3_t gammaCorrect(float3_t c) {
  return float3_t(sqrtf(c.x), sqrtf(c.y), sqrtf(c.z));
}

inline uint32_t RGB32FtoRGB8(float3_t c) {
  int r = (int)(std::min(c.x, 1.f) * 255);
  int g = (int)(std::min(c.y, 1.f) * 255);
  int b = (int)(std::min(c.z, 1.f) * 255);
  return (r << 16) + (g << 8) + b;
}

inline float3_t RGB8toRGB32F(uint32_t c) {
  float s = 1 / 256.0f;
  int r = (c >> 16) & 255;
  int g = (c >> 8) & 255;
  int b = c & 255;
  return float3_t(r * s, g * s, b * s);
}

typedef struct {
  uint32_t dst_width;
  uint32_t dst_height;
  uint64_t dst_addr;

  uint64_t aabb_addr;
  uint64_t shape_addr;
	uint64_t tri_addr;
	uint64_t triEx_addr;
	uint64_t triIdx_addr;
  uint64_t mat_addr;
  uint64_t tex_addr;
	uint64_t bvh_addr;
	uint64_t blas_addr;
  uint64_t tlas_addr;
  uint32_t tlas_root;

  float3_t camera_pos;
	float3_t camera_front;
  float camera_fov;

  uint32_t samples_per_pixel;
  uint32_t max_depth;

  float3_t light_pos;
  float3_t light_color;
  float3_t ambient_color;
  float3_t background_color;
  uint32_t has_sky;

  uint64_t sbt_addr;
} kernel_arg_t;



