
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>
#include "shaders/shader.h"
#include <unordered_map>
#define BLOCK_SIZE 8

ray_t GenerateRay(uint32_t x, uint32_t y, const kernel_arg_t *__UNIFORM__ arg) {
  float x_ndc = (x + 0.5f) / arg->dst_width - 0.5;
  float y_ndc = (y + 0.5f) / arg->dst_height - 0.5;

  float x_vp = x_ndc * arg->viewplane.x;
  float y_vp = y_ndc * arg->viewplane.y;

  auto pt_cam = x_vp * arg->camera_right + y_vp * arg->camera_up + arg->camera_forward;

  auto pt_w = pt_cam + arg->camera_pos;

  auto camera_dir = normalize(pt_w - arg->camera_pos);
  auto camera_pos = arg->camera_pos;

  return ray_t{arg->camera_pos, camera_dir};
}

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto out_ptr = reinterpret_cast<uint32_t *>(arg->dst_addr);
  auto sbt = reinterpret_cast<uint64_t *>(arg->sbt_addr);

  //vx_printf("*** tile: %p\n", miss);
  for (uint32_t ty = 0; ty < BLOCK_SIZE; ++ty) {
    for (uint32_t tx = 0; tx < BLOCK_SIZE; ++tx) {
      uint32_t x = blockIdx.x * BLOCK_SIZE + tx;
      uint32_t y = blockIdx.y * BLOCK_SIZE + ty;
      if (x >= arg->dst_width || y >= arg->dst_height)
        continue;

      float3_t color = float3_t(0, 0, 0);
      for (uint32_t s = 0; s < arg->samples_per_pixel; ++s) {
        auto ray = GenerateRay(x, y, arg);

        uint32_t rayID;
        ray_payload_t payload;
        vortex::rt::traceRay(ray.orig.x, ray.orig.y, ray.orig.z, ray.dir.x, ray.dir.y, ray.dir.z, (uint32_t)(&payload), rayID);

        uint32_t ret;
        while((ret = vortex::rt::getWork()) != 0){
          uint32_t type = __builtin_ctz(ret >> 28);
          uint32_t id = ret & 0x0FFFFFFF;

          auto shader = (shader_t)(sbt[type]);
          shader(id, arg);
        }

        color += payload.color;
      }

      uint32_t globalIdx = x + y * arg->dst_width;
      out_ptr[globalIdx] = RGB32FtoRGB8(color);
    }
  }
}

int main() {
  auto arg = reinterpret_cast<kernel_arg_t *>(csr_read(VX_CSR_MSCRATCH));
  uint32_t grid_dim[2] = {(arg->dst_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (arg->dst_height + BLOCK_SIZE - 1) / BLOCK_SIZE};

  return vx_spawn_threads(2, grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}

//CONFIGS="-DEXT_RTU_ENABLE" ci/blackbox.sh --driver=simx --app=raytracing --cores=1 --args="-mteapot.obj -w40 -h32"