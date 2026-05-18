
#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_raytrace.h>
#include "shaders/shader.h"
#include <unordered_map>

#define BLOCK_SIZE 8
#define PIXELS_PER_TILE 64
#define THREADS_PER_WARP 32
#define BATCH_SIZE 2
#define T_MIN 0.0001f
#define T_MAX LARGE_FLOAT

float fast_rand(uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (float)(seed & 0xFFFFFF) / 16777216.0f;
}

ray_t GenerateRay(uint32_t x, uint32_t y, uint32_t sample_idx, const kernel_arg_t *arg) {
    uint32_t seed = x + y * arg->dst_width + sample_idx * (arg->dst_width * arg->dst_height);

    float jitter_x = fast_rand(seed) - 0.5f;
    float jitter_y = fast_rand(seed) - 0.5f;
    
    // Apply jitter to the screen-space coordinates
    float u = ((float)x + jitter_x) * 2.0f - (float)arg->dst_width;
    u /= (float)arg->dst_height;

    float v = ((float)y + jitter_y) * 2.0f - (float)arg->dst_height;
    v /= (float)arg->dst_height;

    auto FOV = arg->camera_fov;
    auto camera_pos = arg->camera_pos;
    auto camera_front = arg->camera_front;
    auto camera_right = cross(camera_front, float3_t(0.0, 1.0, 0.0));
    auto camera_up = cross(camera_right, camera_front);
    auto camera_dir = normalize(u * camera_right + v * camera_up + FOV * camera_front);

    return ray_t{camera_pos, camera_dir};
}

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  bool *producer_done = (bool*)__local_mem(sizeof(bool));
  *producer_done = 0;
  
  __syncthreads();

  if(vx_warp_id() % 2 == 0){
    // Producer
    
    uint32_t tid = vx_thread_id();

    ray_payload_t payloads[BATCH_SIZE];
    float3_t pixel_accumulator[BATCH_SIZE] = {{0,0,0}, {0,0,0}};

    for (uint32_t s = 0; s < arg->samples_per_pixel; ++s) {
        
        // --- STAGE 1: EMIT BATCH ---
        // Fire the initial primary rays for all payloads in this thread
        for (uint32_t i = 0; i < BATCH_SIZE; ++i) {
            uint32_t thread_idx = tid + (i * THREADS_PER_WARP);
            uint32_t x = blockIdx.x * BLOCK_SIZE + (thread_idx % BLOCK_SIZE);
            uint32_t y = blockIdx.y * BLOCK_SIZE + (thread_idx / BLOCK_SIZE);
            
            if (x < arg->dst_width && y < arg->dst_height) {
              auto ray = GenerateRay(x, y, s, arg);
              payloads[i].origin = ray.orig;
              payloads[i].direction = ray.dir;
              payloads[i].shader_miss_idx = 0;
              payloads[i].shader_record_stride = 1;
              payloads[i].shader_record_offset = 0;
              payloads[i].throughput = {1.0f, 1.0f, 1.0f};
              payloads[i].irradiance = {0.0f, 0.0f, 0.0f};
              payloads[i].bounce = 0;
              payloads[i].stop = false;
              payloads[i].done = false;

              vortex::rt::trace_ray(
                ray.orig.x, 
                ray.orig.y, 
                ray.orig.z,
                ray.dir.x, 
                ray.dir.y, 
                ray.dir.z, 
                T_MIN, T_MAX,
                (uint32_t)(&payloads[i]),
                arg->tlas_addr
              );
            }else{
              payloads[i].stop = true;
              payloads[i].done = true;
            }
        }

        // --- STAGE 2: THE ITERATIVE WAVE ---
        bool any_active = true;
        while (vx_vote_any(any_active)) {
            // Wait for all rays in the warp to finish their current bounce
            for (uint32_t i = 0; i < BATCH_SIZE; ++i) {
                while(!vx_vote_all(payloads[i].done)); 
            }

            any_active = false;
            // Decide if secondary rays are needed
            for (uint32_t i = 0; i < BATCH_SIZE; ++i) {
                if (!payloads[i].stop && payloads[i].bounce < arg->max_depth) {
                    payloads[i].done = false;

                    vortex::rt::trace_ray(
                      payloads[i].origin.x, 
                      payloads[i].origin.y, 
                      payloads[i].origin.z,
                      payloads[i].direction.x, 
                      payloads[i].direction.y, 
                      payloads[i].direction.z, 
                      T_MIN, T_MAX,
                      (uint32_t)(&payloads[i]),
                      arg->tlas_addr
                    );
                    any_active = true;
                }
            }
        }

        // Accumulate results
        for (uint32_t i = 0; i < BATCH_SIZE; ++i) {
            pixel_accumulator[i] += payloads[i].irradiance;
        }
    }

    // Writing back colors
    auto out_ptr = reinterpret_cast<uint32_t *>(arg->dst_addr);
    for (uint32_t i = 0; i < BATCH_SIZE; ++i) {
        uint32_t thread_idx = tid + (i * THREADS_PER_WARP);
        uint32_t x = blockIdx.x * BLOCK_SIZE + (thread_idx % BLOCK_SIZE);
        uint32_t y = blockIdx.y * BLOCK_SIZE + (thread_idx / BLOCK_SIZE);
        
        if (x < arg->dst_width && y < arg->dst_height) {
            // Average the samples
            // vx_printf("(%d, %d)\n", y, x);
            float3_t final_color = pixel_accumulator[i] / (float)arg->samples_per_pixel;
            out_ptr[x + y * arg->dst_width] = RGB32FtoRGB8(final_color);
        }
    }

    *producer_done = 1;
  }else{
    // Consumer
    
    auto sbt = reinterpret_cast<uint64_t *>(arg->sbt_addr);

    while (*producer_done == 0) {
      uint32_t ret = vortex::rt::get_work();
      if (ret != 0) {
        uint32_t type = __builtin_ctz(ret);
        auto shader = (shader_t)(sbt[type]);
        shader(arg);
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