#pragma once
#include <stdint.h>
#include "common.h"

void miss_shader(uint32_t rayID, kernel_arg_t *arg);
void closet_hit_shader(uint32_t rayID, kernel_arg_t *arg);