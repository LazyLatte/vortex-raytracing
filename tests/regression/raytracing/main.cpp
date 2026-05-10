#include "common.h"
#include "tracer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <sstream>

#ifndef ASSETS_PATHS
#define ASSETS_PATHS ""
#endif

const char *kernel_file = "kernel.vxbin";
const char *output_file = "output.ppm";

uint32_t scene_id = 1;

uint32_t dst_width = 64;
uint32_t dst_height = 64;

uint32_t samples_per_pixel = 1;
uint32_t max_depth = 1;

static void show_usage() {
  std::cout << "Vortex Raycaster" << std::endl;
  std::cout << "Usage: [-k kernel] [-n meshes] [-w width] [-h height]" << std::endl;
  std::cout << "       [-m model] [-s samples] [-d depth] [-f vfov] [-z zoom] [-c]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int opt;
  while ((opt = getopt(argc, argv, "k:n:m:w:h:s:f:z:0d:o:c")) != -1) {
    switch (opt) {
      case 'k':
        kernel_file = optarg;
        break;
      case 'n':
        // mesh_count = atoi(optarg);
        break;
      case 'm':
        scene_id = atoi(optarg);
        break;
      case 'w':
        dst_width = atoi(optarg);
        break;
      case 'h':
        dst_height = atoi(optarg);
        break;
      case 'f':
        // camera_fvf = atof(optarg);
        break;
      case 'z':
        // camera_zoom = atof(optarg);
        break;
      case 's':
        samples_per_pixel = atoi(optarg);
        break;
      case 'd':
        max_depth = atoi(optarg);
        break;
      case 'o':
        output_file = optarg;
        break;
      default:
        show_usage();
        exit(-1);
    }
  }

  // Handle any remaining non-option arguments if needed
  if (optind < argc) {
    std::cout << "Non-option arguments: ";
    while (optind < argc) {
      std::cout << argv[optind++] << " ";
    }
    std::cout << std::endl;
    show_usage();
    exit(-1);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  Tracer tracer(dst_width, dst_height, samples_per_pixel, max_depth);
  RT_CHECK(tracer.init(kernel_file, scene_id));
  RT_CHECK(tracer.setup());
  RT_CHECK(tracer.run(output_file));

  return 0;
}
