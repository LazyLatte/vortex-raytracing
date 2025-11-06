#pragma once
#include <cstdint>
#define LARGE_FLOAT 1e30f

namespace vortex {
    struct Hit {
        float dist = LARGE_FLOAT, bx, by, bz;
        uint32_t blasIdx, triIdx;
    };

    struct Ray {
        float ro_x, ro_y, ro_z, rd_x, rd_y, rd_z;
    };

    struct RayBuffer {
        uint32_t id;
        Ray ray;
        
        Hit hit;
        RayBuffer(){}
    };

}