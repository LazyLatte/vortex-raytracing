
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>
#include "shaders/shader.h"
#include <unordered_map>

#define BLOCK_SIZE 8
#define PIXELS_PER_TILE 64
#define THREADS_PER_WARP 32

// ray_t GenerateRay(uint32_t x, uint32_t y, const kernel_arg_t *__UNIFORM__ arg) {
//   float x_ndc = (x + 0.5f) / arg->dst_width - 0.5;
//   float y_ndc = (y + 0.5f) / arg->dst_height - 0.5;

//   float x_vp = x_ndc * arg->viewplane.x;
//   float y_vp = y_ndc * arg->viewplane.y;

//   auto pt_cam = x_vp * arg->camera_right + y_vp * arg->camera_up + arg->camera_forward;

//   auto pt_w = pt_cam + arg->camera_pos;

//   auto camera_dir = normalize(pt_w - arg->camera_pos);

//   return ray_t{arg->camera_pos, camera_dir};
// }

ray_t GenerateRay(uint32_t x, uint32_t y, const kernel_arg_t *__UNIFORM__ arg) {
  auto pos = float3_t(0.0, 100.0, 0.0);
  auto front = float3_t(1.0, 0.0, 0.0);
  float FOV = 1.0;
  float u = (x * 2.0 - arg->dst_width) / arg->dst_height;
  float v = (y * 2.0 - arg->dst_height) / arg->dst_height;

  auto right = cross(front, float3_t(0.0, 1.0, 0.0));
  auto up = cross(right, front);
  auto dir = normalize(u * right + v * up + FOV * front);
  return ray_t{pos, dir};
}

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  //vx_printf("*** tile: %d\n", *producers_finished);
  auto sbt = reinterpret_cast<uint64_t *>(arg->sbt_addr);

  volatile bool *local_done = (volatile bool*)__local_mem(sizeof(bool));
  *local_done = 0;
  
  __syncthreads();

  if(vx_warp_id() % 2 == 0){
    // Producer
    ray_payload_t payloads[2];
    payloads[0].done = true;
    payloads[1].done = true;
    uint32_t tid = vx_thread_id();
    for (uint32_t i = tid; i < PIXELS_PER_TILE; i += THREADS_PER_WARP) {
      uint32_t tx = i % BLOCK_SIZE;
      uint32_t ty = i / BLOCK_SIZE;
      uint32_t x = blockIdx.x * BLOCK_SIZE + tx;
      uint32_t y = blockIdx.y * BLOCK_SIZE + ty;
      
      if (x < arg->dst_width && y < arg->dst_height){
        //float3_t color = float3_t(0, 0, 0);
        
        payloads[i/THREADS_PER_WARP].done = false;

        for (uint32_t s = 0; s < arg->samples_per_pixel; ++s) {
          auto ray = GenerateRay(x, y, arg);

          uint32_t rayID;
          vortex::rt::traceRay(
            ray.orig.x, ray.orig.y, ray.orig.z, 
            ray.dir.x, ray.dir.y, ray.dir.z, 
            (uint32_t)(&payloads[i/THREADS_PER_WARP]), 
            rayID
          );
          
          //color += payload.color;
        }
      }
    }

    // Poll
    while(!vx_vote_all(payloads[0].done && payloads[1].done)){
      uint32_t ret = vortex::rt::getWork();
      if (ret != 0) {
        uint32_t type = __builtin_ctz(ret >> 28);
        uint32_t id = ret & 0x0FFFFFFF;
        auto shader = (shader_t)(sbt[type]);
        shader(id, arg);
      }
    }
    
    // Writing back colors
    auto out_ptr = reinterpret_cast<uint32_t *>(arg->dst_addr);

    for (uint32_t i = tid; i < PIXELS_PER_TILE; i += THREADS_PER_WARP) {
      uint32_t tx = i % BLOCK_SIZE;
      uint32_t ty = i / BLOCK_SIZE;
      uint32_t x = blockIdx.x * BLOCK_SIZE + tx;
      uint32_t y = blockIdx.y * BLOCK_SIZE + ty;
      if (x < arg->dst_width && y < arg->dst_height) {
        //vx_printf("(%d, %d)\n", y, x);
        out_ptr[x + y * arg->dst_width] = RGB32FtoRGB8(payloads[i/THREADS_PER_WARP].color);
      }
    }

    vx_fence();
    *local_done = 1;
  }else{

    // Consumer
    while (*local_done == 0) {
      uint32_t ret = vortex::rt::getWork();
      if (ret != 0) {
        uint32_t type = __builtin_ctz(ret >> 28);
        uint32_t id = ret & 0x0FFFFFFF;
        auto shader = (shader_t)(sbt[type]);
        shader(id, arg);
      }
    }

  }

  __syncthreads();
}

int main() {
  auto arg = reinterpret_cast<kernel_arg_t *>(csr_read(VX_CSR_MSCRATCH));
  uint32_t grid_dim[2] = {(arg->dst_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (arg->dst_height + BLOCK_SIZE - 1) / BLOCK_SIZE};
  uint32_t block_dim[2] = {BLOCK_SIZE, BLOCK_SIZE};
  return vx_spawn_threads(2, grid_dim, block_dim, (vx_kernel_func_cb)kernel_body, arg);
}

//CONFIGS="-DEXT_RTU_ENABLE" ci/blackbox.sh --driver=simx --app=raytracing --cores=1 --args="-mteapot.obj -w40 -h32"
//make -C raytracing clean