#include "tracer.h"
#define __UNIFORM__
#include <iostream>
#include <fstream>
#include <sstream>

static void write_ppm(const uint8_t* output, uint32_t width, uint32_t height, const char *output_file) {
  std::ofstream ofs(output_file, std::ios::binary);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open output file: " << output_file << std::endl;
    std::abort();
  }
  ofs << "P3\n" << width << " " << height << "\n255\n";

  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      size_t idx = (height - 1 - y) * width + x;
      float r = output[idx * 4 + 0];
      float g = output[idx * 4 + 1];
      float b = output[idx * 4 + 2];
      ofs << b << " " << g << " " << r << "\n";
    }
  }
  std::cout << "Image saved to: " << output_file << std::endl;
}

Tracer::Tracer(
  uint32_t dst_width,
  uint32_t dst_height,
  uint32_t samples_per_pixel,
  uint32_t max_depth)
  : dst_width_(dst_width)
  , dst_height_(dst_height)
{
  kernel_arg_.dst_width = dst_width_;
  kernel_arg_.dst_height = dst_height_;
  kernel_arg_.samples_per_pixel = samples_per_pixel;
  kernel_arg_.max_depth = max_depth;
}

Tracer::~Tracer() {
  // free allocated objects
  delete scene_;

  // free buffers
  vx_mem_free(output_buffer_);
  vx_mem_free(args_buffer_);
  vx_mem_free(krnl_buffer_);
  vx_mem_free(triBuffer_);
  vx_mem_free(triExBuffer_);
  vx_mem_free(matBuffer_);
  vx_mem_free(texBuffer_);
  vx_mem_free(tlasBuffer_);
  vx_mem_free(blasBuffer_);
  vx_mem_free(bvhBuffer_);

  vx_mem_free(shapeBuffer_);
  vx_mem_free(sbtBuffer_);
  vx_mem_free(rMiss_buffer_);
  vx_mem_free(rchit_buffer_);
  vx_mem_free(rint_buffer_);
  vx_mem_free(rahit_buffer_);
  // close device
  vx_dev_close(device_);
}

int Tracer::init(const char *kernel_file, uint32_t scene_id) {
  // create scene
  const std::pair<std::string, std::function<Scene* (void)>> target_scene = SceneList::AllScenes[scene_id];

  std::string scene_name = target_scene.first;
  std::function<Scene* (void)> CreateScene =  target_scene.second;
  
  scene_ = CreateScene();
  RT_CHECK(scene_->init());

  RT_CHECK(vx_dev_open(&device_));

  // upload kernel
  // vx_mem_address(krnl_buffer_) get base address
  RT_CHECK(vx_upload_kernel_file(device_, kernel_file, &krnl_buffer_));
  RT_CHECK(vx_upload_kernel_file(device_, "miss.vxbin", &rMiss_buffer_));
  RT_CHECK(vx_upload_kernel_file(device_, "closet-hit.vxbin", &rchit_buffer_));
  RT_CHECK(vx_upload_kernel_file(device_, "any-hit.vxbin", &rahit_buffer_));
  RT_CHECK(vx_upload_kernel_file(device_, "intersection.vxbin",  &rint_buffer_));

  // allocate tri buffer
  RT_CHECK(vx_mem_alloc(device_, scene_->tri_buf().size() * sizeof(tri_t), VX_MEM_READ, &triBuffer_));
  RT_CHECK(vx_mem_address(triBuffer_, &kernel_arg_.tri_addr));

  // allocate triEx buffer
  RT_CHECK(vx_mem_alloc(device_, scene_->triEx_buf().size() * sizeof(tri_ex_t), VX_MEM_READ, &triExBuffer_));
  RT_CHECK(vx_mem_address(triExBuffer_, &kernel_arg_.triEx_addr));

  // allocate tlas buffer
  RT_CHECK(vx_mem_alloc(device_, scene_->tlas_nodes().size() * sizeof(cwbvh_node_t), VX_MEM_READ, &tlasBuffer_));
  RT_CHECK(vx_mem_address(tlasBuffer_, &kernel_arg_.tlas_addr));

  // allocate inst buffer
  RT_CHECK(vx_mem_alloc(device_, scene_->blas_nodes().size() * sizeof(blas_node_t), VX_MEM_READ, &blasBuffer_));
  RT_CHECK(vx_mem_address(blasBuffer_, &kernel_arg_.blas_addr));

  // allocate bvh buffer
  RT_CHECK(vx_mem_alloc(device_, scene_->cwbvh_nodes().size() * sizeof(cwbvh_node_t), VX_MEM_READ, &bvhBuffer_));
  RT_CHECK(vx_mem_address(bvhBuffer_, &kernel_arg_.bvh_addr));

  // allocate AABB buffer
  if(scene_->aabb_buf().size() > 0){
    RT_CHECK(vx_mem_alloc(device_, scene_->aabb_buf().size() * sizeof(AABB), VX_MEM_READ, &aabbBuffer_));
    RT_CHECK(vx_mem_address(aabbBuffer_, &kernel_arg_.aabb_addr));
  }

  // allocate shape buffer
  if (scene_->shape_buf().size() > 0) {
    RT_CHECK(vx_mem_alloc(device_, scene_->shape_buf().size() * sizeof(shape_t), VX_MEM_READ, &shapeBuffer_));
    RT_CHECK(vx_mem_address(shapeBuffer_, &kernel_arg_.shape_addr));
  }
  
  // allocate mat buffer
  if(scene_->mat_buf().size() > 0){
    RT_CHECK(vx_mem_alloc(device_, scene_->mat_buf().size() * sizeof(material_info_t), VX_MEM_READ, &matBuffer_));
    RT_CHECK(vx_mem_address(matBuffer_, &kernel_arg_.mat_addr));
  }

  // allocate tex buffer
  if(scene_->tex_buf().size() > 0){
    RT_CHECK(vx_mem_alloc(device_, scene_->tex_buf().size(), VX_MEM_READ, &texBuffer_));
    RT_CHECK(vx_mem_address(texBuffer_, &kernel_arg_.tex_addr));
  }

  // allocate output buffer
  RT_CHECK(vx_mem_alloc(device_, dst_width_ * dst_height_ * sizeof(uint32_t), VX_MEM_WRITE, &output_buffer_));
  RT_CHECK(vx_mem_address(output_buffer_, &kernel_arg_.dst_addr));

  // allocate sbt
  RT_CHECK(vx_mem_alloc(device_, sizeof(uint64_t) * 4, VX_MEM_READ, &sbtBuffer_));
  RT_CHECK(vx_mem_address(sbtBuffer_, &kernel_arg_.sbt_addr));
  
  return 0;
}

int Tracer::setup() {

  // build the scene
  scene_->build();

  // setup Camera
  {
    kernel_arg_.camera_pos = scene_->camera_pos;
    kernel_arg_.camera_front = scene_->camera_front;
    kernel_arg_.camera_fov = scene_->camera_fov;
  }

  // setup lighting
  {
    kernel_arg_.light_pos = scene_->light_pos;
    kernel_arg_.light_color = scene_->light_color;
    kernel_arg_.ambient_color = scene_->ambient_color;
    kernel_arg_.background_color = scene_->background_color;
  }

  // upload tri data
  RT_CHECK(vx_copy_to_dev(triBuffer_, scene_->tri_buf().data(), 0, scene_->tri_buf().size() * sizeof(tri_t)));

  // upload triEx data
  RT_CHECK(vx_copy_to_dev(triExBuffer_, scene_->triEx_buf().data(), 0, scene_->triEx_buf().size() * sizeof(tri_ex_t)));

  // upload tlas data
  RT_CHECK(vx_copy_to_dev(tlasBuffer_, scene_->cwtlas_nodes().data(), 0, scene_->cwtlas_nodes().size() * sizeof(cwbvh_node_t)));

  // upload inst data
  RT_CHECK(vx_copy_to_dev(blasBuffer_, scene_->blas_nodes().data(), 0, scene_->blas_nodes().size() * sizeof(blas_node_t)));

  // upload bvh data
  RT_CHECK(vx_copy_to_dev(bvhBuffer_, scene_->cwbvh_nodes().data(), 0, scene_->cwbvh_nodes().size() * sizeof(cwbvh_node_t)));

  // upload AABB data
  if(scene_->aabb_buf().size() > 0)
    RT_CHECK(vx_copy_to_dev(aabbBuffer_, scene_->aabb_buf().data(), 0, scene_->aabb_buf().size() * sizeof(AABB)));

  // upload shape data
  if (scene_->shape_buf().size() > 0)
    RT_CHECK(vx_copy_to_dev(shapeBuffer_, scene_->shape_buf().data(), 0, scene_->shape_buf().size() * sizeof(shape_t)));

  // upload mat data
  if(scene_->mat_buf().size() > 0)
    RT_CHECK(vx_copy_to_dev(matBuffer_, scene_->mat_buf().data(), 0, scene_->mat_buf().size() * sizeof(material_info_t)));

  // upload tex data
  if(scene_->tex_buf().size() > 0)
    RT_CHECK(vx_copy_to_dev(texBuffer_, scene_->tex_buf().data(), 0, scene_->tex_buf().size()));

  // upload sbt
  uint64_t tmp_sbt[4];
  RT_CHECK(vx_mem_address(rMiss_buffer_, &tmp_sbt[0]));
  RT_CHECK(vx_mem_address(rchit_buffer_, &tmp_sbt[1]));
  RT_CHECK(vx_mem_address(rint_buffer_, &tmp_sbt[2]));
  RT_CHECK(vx_mem_address(rahit_buffer_, &tmp_sbt[3]));
  RT_CHECK(vx_copy_to_dev(sbtBuffer_, tmp_sbt, 0, sizeof(uint64_t) * 4));

  RT_CHECK(vx_dcr_write(device_, 0x00000006, (uint32_t)(kernel_arg_.tlas_addr)));
  RT_CHECK(vx_dcr_write(device_, 0x00000007, (uint32_t)(kernel_arg_.blas_addr)));
  RT_CHECK(vx_dcr_write(device_, 0x00000008, (uint32_t)(kernel_arg_.bvh_addr)));
  RT_CHECK(vx_dcr_write(device_, 0x00000009, (uint32_t)(kernel_arg_.tri_addr)));
  RT_CHECK(vx_dcr_write(device_, 0x0000000A, (uint32_t)(kernel_arg_.aabb_addr)));
  
  return 0;
}

int Tracer::run(const char *output_file) {
  std::vector<uint8_t> h_output(dst_width_ * dst_height_ * sizeof(uint32_t));

  std::cout << "Begin rendering to " << dst_width_ << "x" << dst_height_ << " framebuffer." << std::endl;

  // upload kernel arguments
  RT_CHECK(vx_upload_bytes(device_, &kernel_arg_, sizeof(kernel_arg_t), &args_buffer_));

  // start kernel execution
  RT_CHECK(vx_start(device_, krnl_buffer_, args_buffer_));

  // wait for the kernel to finish
  RT_CHECK(vx_ready_wait(device_, VX_MAX_TIMEOUT));

  // download the output buffer
  RT_CHECK(vx_copy_from_dev(h_output.data(), output_buffer_, 0, h_output.size()));

  // write the output to a PPM file
  write_ppm(h_output.data(), dst_width_, dst_height_, output_file);

  return 0;
}