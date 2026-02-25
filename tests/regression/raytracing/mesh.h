#pragma once

#include "bvh.h"
#include "surface.h"
#include "common.h"
#include <iostream>

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
  Mesh(const char *objFile);
  ~Mesh();

  const std::vector<tri_t>& tri() const { return tri_; }
  const std::vector<tri_ex_t>& triEx() const { return triEx_; }
  const std::vector<Surface*>& textures() const { return textures_; }
  const std::vector<material_info_t>& materials() const { return materials_; }
private:
  std::vector<tri_t> tri_;
  std::vector<tri_ex_t> triEx_;
  std::vector<Surface*> textures_;
  std::vector<material_info_t> materials_;
};
