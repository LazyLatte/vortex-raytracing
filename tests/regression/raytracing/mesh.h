#pragma once

#include "bvh.h"
#include "surface.h"
#include "common.h"
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <cassert>

struct obj_mesh_t {
  struct vert_t {
    uint32_t p;
    uint32_t n;
    uint32_t t;
  };

  struct face_t {
    uint32_t v[3];
    int material_id;
  };

  std::vector<float3_t> positions;
  std::vector<float3_t> normals;
  std::vector<float2_t> texcoords;
  std::vector<vert_t>   vertices;
  std::vector<face_t>   faces;
};

struct material_textures_t {
  std::string ambient_texname;
  std::string diffuse_texname;
  std::string specular_texname;
  std::string specular_highlight_texname;
  std::string bump_texname;
  std::string displacement_texname;
  std::string alpha_texname;
  std::string reflection_texname;
};

// 3D object container
class Mesh {
public:
  explicit Mesh(const std::filesystem::path& objFile, uint32_t geometry_idx, bool opaque = true);
  ~Mesh() = default;

  Mesh(const Mesh&) = delete;
  Mesh& operator=(const Mesh&) = delete;

  Mesh(Mesh&&) = default;
  Mesh& operator=(Mesh&&) = default;

  const std::vector<tri_t>& tri() const { return tri_; }
  const std::vector<tri_ex_t>& triEx() const { return triEx_; }
  const std::vector<material_info_t>& materials() const { return materials_; }
  std::vector<Surface*> textures() const;

  bool isOpaque() const { return opaque_; };
  uint32_t getGeometryIndex() const { return geometry_idx; }

private:
  std::vector<tri_t> tri_;
  std::vector<tri_ex_t> triEx_;
  std::vector<material_info_t> materials_;
  std::vector<std::unique_ptr<Surface>> textures_;

  uint32_t geometry_idx;
  bool opaque_;
};
