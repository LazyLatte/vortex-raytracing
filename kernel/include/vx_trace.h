// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vx_intrinsics.h>

namespace vortex {
namespace rt {

static __attribute__((always_inline)) void traceRay(
    float ro_x, 
    float ro_y, 
    float ro_z, 
    float rd_x, 
    float rd_y, 
    float rd_z, 
    uint32_t& rayID
) {
    
    register float ox __asm__("f10") = ro_x;
    register float oy __asm__("f11") = ro_y;
    register float oz __asm__("f12") = ro_z;

    register float dx __asm__("f13") = rd_x;
    register float dy __asm__("f14") = rd_y;
    register float dz __asm__("f15") = rd_z;

    register uint32_t ret __asm__("x6");

    __asm__ volatile (
        ".insn r %[insn], 0, 3, %[rd_t], %[rs_base], x0"
        : [rd_t] "=r"(ret)
        : [insn] "i"(RISCV_CUSTOM0), [rs_base] "f"(ox),
          "f"(oy), "f"(oz), "f"(dx), "f"(dy), "f"(dz)
    );

    rayID = ret;
}



inline int getWork() {
    int ret;
    __asm__ volatile (".insn r %1, 1, 3, %0, x0, x0" : "=r"(ret) : "i"(RISCV_CUSTOM0));
    return ret;
}

inline int getAttr(uint32_t rayID, uint32_t idx) {
    int ret;
    __asm__ volatile (".insn r %1, 2, 3, %0, %2, %3" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(rayID), "r"(idx));
    return ret;
}

static __attribute__((always_inline)) void setColor(uint32_t rayID, float red, float green, float blue) {
    
    register float r __asm__("f7") = red;
    register float g __asm__("f8") = green;
    register float b __asm__("f9") = blue;

    __asm__ volatile (
        ".insn r %[insn], 4, 3, x0, %[id], %[rs_base]"
        :: [insn] "i"(RISCV_CUSTOM0), [id] "r"(rayID), [rs_base] "f"(r), "f"(g), "f"(b)
    );
}

inline int getColor(uint32_t rayID, uint32_t idx) {
    int ret;
    __asm__ volatile (".insn r %1, 5, 3, %0, %2, %3" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(rayID), "r"(idx));
    return ret;
}

} 
} 
