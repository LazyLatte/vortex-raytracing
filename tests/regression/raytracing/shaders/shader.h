#pragma once
#include <stdint.h>
#include "../common.h"

//relocatable binary!!!
typedef void (*shader_t)(uint32_t rayID, kernel_arg_t *arg);

struct ray_payload_t {
    float3_t color;
    uint32_t flag;

    uint32_t bounce = 0;
};