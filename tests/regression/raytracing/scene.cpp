#include "scene.h"
#include "treelet.h"
Scene::Scene(const std::vector<Mesh*> &meshes)
  : meshes_(meshes), bvh_list_(meshes.size(), nullptr)
{}

Scene::~Scene() {
  for (auto mesh : meshes_) {
    delete mesh;
  }
  for (auto bvh : bvh_list_) {
    delete bvh;
  }
  delete tlas_;
}

int Scene::init() {
  // estimate buffers size
  uint32_t num_tris = 0;
  uint64_t total_texture_size = 0;
  uint32_t total_materials = 0;
  //uint64_t texture_size = 0;
  for (auto mesh : meshes_) {
    num_tris += mesh->tri().size();
    total_materials += mesh->materials().size();
    //texture_size += mesh->texture()->size();
    for (auto tex : mesh->textures()) {
      if (tex) total_texture_size += tex->size();
    }
  }

  // allocate buffers
  tex_buf_.resize(total_texture_size);
  mat_buf_.resize(total_materials);
  tri_buf_.resize(num_tris);
  triEx_buf_.resize(num_tris);
  triIdx_buf_.resize(num_tris);
  centroids_.resize(num_tris);
  bvh_nodes_.resize(num_tris * 2);
  cwbvh_nodes_.resize(num_tris * 2);
  blas_nodes_.resize(meshes_.size());

  aabb_buf_.resize(num_tris);

  // create BVH objects
  uint32_t bvh_offset = 0;
  uint32_t tri_offset = 0;
  uint64_t tex_offset = 0;
  uint32_t mat_offset = 0;

  for (uint32_t i = 0; i < meshes_.size(); ++i) {
    auto mesh = meshes_.at(i);

    // copy mesh buffers
    memcpy(tri_buf_.data() + tri_offset, mesh->tri().data(), mesh->tri().size() * sizeof(tri_t));

    auto* triEx_ptr = triEx_buf_.data() + tri_offset;
    memcpy(triEx_ptr, mesh->triEx().data(), mesh->tri().size() * sizeof(tri_ex_t));
    for (uint32_t j = 0; j < mesh->tri().size(); ++j) {
      triEx_ptr[j].texId += mat_offset;
    }

    std::vector<uint64_t> mesh_tex_offsets;
    for (auto tex : mesh->textures()) {
      if (tex) {
        mesh_tex_offsets.push_back(tex_offset);
        memcpy(tex_buf_.data() + tex_offset, tex->pixels(), tex->size());
        tex_offset += tex->size();
      } else {
        mesh_tex_offsets.push_back(0); // No texture
      }
    }
    //memcpy(tex_buf_.data() + tex_offset, mesh->texture()->pixels(), mesh->texture()->size());
    for (uint32_t m = 0; m < mesh->materials().size(); ++m) {
      auto mat = mesh->materials()[m];
      if (mat.diffuse_tex_id >= 0) {
        mat.tex_offset = mesh_tex_offsets[mat.diffuse_tex_id];
        mat.tex_width = mesh->textures()[mat.diffuse_tex_id]->width();
        mat.tex_height = mesh->textures()[mat.diffuse_tex_id]->height();
      }
      mat_buf_[mat_offset + m] = mat;
    }

    // precompute triangle indices & centroids
    for (uint32_t j = 0; j < mesh->tri().size(); ++j) {
      auto &tri = mesh->tri().at(j);
      triIdx_buf_[tri_offset + j] = tri_offset + j;
      centroids_[tri_offset + j] = (tri.v0 + tri.v1 + tri.v2) / 3;
    }

    // setup blas node
    auto &blas_node = blas_nodes_.at(i);
    blas_node.bvh_offset = bvh_offset;
    blas_node.mat_offset = mat_offset;
    blas_node.transform = mat4_t::Identity();
    blas_node.invTransform = mat4_t::Identity();
    // create BVH

    auto bvh = new BVH(tri_buf_.data(), centroids_.data(), mesh->tri().size(), bvh_nodes_.data() + bvh_offset, triIdx_buf_.data() + tri_offset);

    this->compressNodes(tri_offset);

    // update offsets
    bvh_offset += bvh->nodeCount();
    tri_offset += mesh->tri().size();
    mat_offset += mesh->materials().size();
    bvh_list_[i] = bvh;
  }

  this->linearizeData();
  
  // create TLAS
  tlas_ = new TLAS(bvh_list_, blas_nodes_.data());
  
  return 0;
}

void Scene::compressNodes(uint32_t tri_offset){
  for(int i=0; i<bvh_nodes_.size(); i++){
    const bvh_node_t& node = bvh_nodes_[i];
    cwbvh_node_t& cwNode = cwbvh_nodes_[i];

    cwNode.leftFirst = node.leftFirst; 

    if(!node.isLeaf()){
      cwNode.type = BVH_INTERNAL;
      cwNode.internal.origin = node.aabbMin;

      cwNode.ex = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.x - node.aabbMin.x) / 255.0f)));
      cwNode.ey = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.y - node.aabbMin.y) / 255.0f)));
      cwNode.ez = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.z - node.aabbMin.z) / 255.0f)));

      for(int k=0; k<BVH_WIDTH; k++){
        child_data_t qChild;

        if(k < node.childCount){
          const bvh_node_t& childNode = bvh_nodes_[node.leftFirst + k];

          qChild.meta = 1; //Do we need meta info, or just a single valid bit?

          qChild.qaabb[0] = static_cast<uint8_t>(std::floor((childNode.aabbMin.x - cwNode.internal.origin.x) / std::exp2f(static_cast<float>(cwNode.ex))));
          qChild.qaabb[1] = static_cast<uint8_t>(std::floor((childNode.aabbMin.y - cwNode.internal.origin.y) / std::exp2f(static_cast<float>(cwNode.ey))));
          qChild.qaabb[2] = static_cast<uint8_t>(std::floor((childNode.aabbMin.z - cwNode.internal.origin.z) / std::exp2f(static_cast<float>(cwNode.ez))));

          qChild.qaabb[3] = static_cast<uint8_t>(std::ceil((childNode.aabbMax.x - cwNode.internal.origin.x) / std::exp2f(static_cast<float>(cwNode.ex))));
          qChild.qaabb[4] = static_cast<uint8_t>(std::ceil((childNode.aabbMax.y - cwNode.internal.origin.y) / std::exp2f(static_cast<float>(cwNode.ey))));
          qChild.qaabb[5] = static_cast<uint8_t>(std::ceil((childNode.aabbMax.z - cwNode.internal.origin.z) / std::exp2f(static_cast<float>(cwNode.ez))));

        }else{
          qChild.meta = 0;
        }
        cwNode.internal.children[k] = qChild;
      }
    }else{
      cwNode.type = TRIANGLE_LEAF;
      //cwNode.type = PROCEDURAL_LEAF;
      cwNode.leaf.flags = OPAQUE;
      //cwNode.leaf.flags = NON_OPAQUE;
      cwNode.leaf.primCount = node.triCount;
      cwNode.leftFirst += tri_offset;
    }
  }
}

void Scene::linearizeData(){
  const uint32_t num_tris = triIdx_buf_.size();
  std::vector<tri_t> sortedTriData(num_tris);
  std::vector<tri_ex_t> sortedTriEx(num_tris);
  std::vector<float3_t> sortedCentroids(num_tris);

  for (uint32_t i = 0; i < num_tris; i++) {
    uint32_t triIdx = triIdx_buf_[i];
    sortedTriData[i] = tri_buf_[triIdx];
    sortedTriEx[i] = triEx_buf_[triIdx];
    sortedCentroids[i] = centroids_[triIdx];

    aabb_buf_[i] = AABB(tri_buf_[triIdx]);
  }

  std::copy(sortedTriData.begin(), sortedTriData.end(), tri_buf_.data());
  std::copy(sortedTriEx.begin(), sortedTriEx.end(), triEx_buf_.data());
  std::copy(sortedCentroids.begin(), sortedCentroids.end(), centroids_.data());
}

void Scene::applyTransform(const mat4_t &transform) {
  for (auto &node : blas_nodes_) {
    node.applyTransform(transform);
  }
}

void Scene::build() {
  // build TLAS
  tlas_->build();
}