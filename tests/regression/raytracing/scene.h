#pragma once

#include "bvh.h"
#include "mesh.h"

// Manages all meshes in the scene
class Scene {
public:
  Scene(const std::vector<Mesh*> &meshes);
  ~Scene();

  int init();

  void applyTransform(const mat4_t &transform);
  void build();

  const auto &tlas_nodes() const { return tlas_->nodes(); }
  const auto &cwtlas_nodes() const { return cwtlas_nodes_; }
  //uint32_t tlas_root() const { return tlas_->rootIndex(); }

  const auto &blas_nodes() const { return blas_nodes_; }
  const auto &cwbvh_nodes() const { return cwbvh_nodes_; }
  const auto &aabb_buf() const { return aabb_buf_; }
  const auto &tri_buf() const { return tri_buf_; }
  const auto &triEx_buf() const { return triEx_buf_; }
  const auto &triIdx_buf() const { return triIdx_buf_; }
  const auto &tex_buf() const { return tex_buf_; }
  const auto &mat_buf() const { return mat_buf_; }

  float3_t camera_pos;
  float3_t camera_front;
  float camera_fov;

  float3_t light_pos;
  float3_t light_color;
  float3_t ambient_color;
  float3_t background_color;

private:
  void compressTLAS();
  void compressBLAS(BVH* bvh, uint32_t tri_offset, uint32_t geometryIndex, bool opaque);
  void linearizeData();

  std::vector<Mesh *> meshes_;
  std::vector<BVH *> bvh_list_;
  std::vector<float3_t> centroids_;
  TLAS *tlas_;

  std::vector<blas_node_t> blas_nodes_;
  std::vector<bvh_node_t> bvh_nodes_;
  std::vector<cwbvh_node_t> cwbvh_nodes_;
  std::vector<cwbvh_node_t> cwtlas_nodes_;
  std::vector<tri_t> tri_buf_;
  std::vector<tri_ex_t> triEx_buf_;
  std::vector<uint32_t> triIdx_buf_;
  std::vector<uint8_t> tex_buf_;
  std::vector<material_info_t> mat_buf_;

  std::vector<AABB> aabb_buf_;
};
