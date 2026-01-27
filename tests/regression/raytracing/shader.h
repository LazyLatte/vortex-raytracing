#pragma once
#include <stdint.h>
#include "common.h"

typedef void (*shader_t)(uint32_t rayID, kernel_arg_t *arg);

void miss_shader(uint32_t rayID, kernel_arg_t *arg);
void closet_hit_shader(uint32_t rayID, kernel_arg_t *arg);
void intersection_shader(uint32_t rayID, kernel_arg_t *arg);
void any_hit_shader(uint32_t rayID, kernel_arg_t *arg);