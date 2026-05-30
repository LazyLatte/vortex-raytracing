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

  // Creates a single procedural sphere primitive encoded as one tri_t entry.
  // v0 = center−r, v1 = center, v2 = center+r gives exact BVH bounds and centroid.
  static Mesh* CreateSphere(float3_t center, float radius, float3_t color, uint32_t geometry_idx);

  Mesh(const Mesh&) = delete;
  Mesh& operator=(const Mesh&) = delete;

  Mesh(Mesh&&) = default;
  Mesh& operator=(Mesh&&) = default;

  const std::vector<tri_t>& tri() const { return tri_; }
  const std::vector<tri_ex_t>& triEx() const { return triEx_; }
  const std::vector<material_info_t>& materials() const { return materials_; }
  const std::vector<shape_t>& shapes() const { return shapes_; }
  const std::vector<AABB>& aabbs() const { return aabbs_; }
  std::vector<Surface*> textures() const;

  bool isOpaque() const { return opaque_; };
  bool isProcedural() const { return procedural_; }
  uint32_t getGeometryIndex() const { return geometry_idx; }

  void setAllDiffuse(float3_t color) {
    for (auto& m : materials_) {
      m.diffuse = color;
      m.diffuse_tex_id = -1;
      m.tex_width = 0;
      m.tex_height = 0;
    }
  }

  void setAllReflectivity(float r) {
    for (auto& m : materials_)
      m.reflectivity = r;
  }

  void setAllFuzz(float fuzz) {
    for (auto& m : materials_)
      m.shininess = fuzz;
  }

  void setAllIOR(float ior) {
    for (auto& m : materials_)
      m.ior = ior;
  }

  void setAllMaterialModel(int model) {
    for (auto& m : materials_)
      m.illum = model;
  }

  void setTransform(const mat4_t& T){
    this->transform = T;
  }

  mat4_t getTransform() const {return this->transform; }

private:
  std::vector<tri_t> tri_;
  std::vector<tri_ex_t> triEx_;
  std::vector<material_info_t> materials_;
  std::vector<std::unique_ptr<Surface>> textures_;
  std::vector<shape_t> shapes_;
  std::vector<AABB> aabbs_;

  // Private ctor used by CreateSphere.
  Mesh(tri_t tri, tri_ex_t triEx, material_info_t mat, uint32_t geometry_idx);

  mat4_t transform;
  uint32_t geometry_idx;
  bool opaque_    = true;
  bool procedural_ = false;
};
