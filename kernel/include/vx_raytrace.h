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

static __attribute__((always_inline)) void trace_ray(
    float ro_x, 
    float ro_y, 
    float ro_z, 
    float rd_x, 
    float rd_y, 
    float rd_z, 
    float tmin,
    float tmax,
    uint32_t payload_addr,
    uint32_t tlas_addr
) {
    register float ox __asm__("f11") = ro_x;
    register float oy __asm__("f12") = ro_y;
    register float oz __asm__("f13") = ro_z;

    register float dx __asm__("f14") = rd_x;
    register float dy __asm__("f15") = rd_y;
    register float dz __asm__("f16") = rd_z;

    register float rs_tmin __asm__("f17") = tmin;
    register float rs_tmax __asm__("f18") = tmax;

    register uint32_t rs_payload_addr __asm__("x19") = payload_addr;
    register uint32_t rs_tlas_addr __asm__("x20") = tlas_addr;

    __asm__ volatile (
        ".insn r %[insn], 0, 3, x0, %[rs_ray], %[rs_addrs]"
        :: [insn] "i"(RISCV_CUSTOM0)
        , [rs_ray] "f"(ox), "f"(oy), "f"(oz), "f"(dx), "f"(dy), "f"(dz), "f"(rs_tmin), "f"(rs_tmax)
        , [rs_addrs] "r"(rs_payload_addr), "r"(rs_tlas_addr)
    );
}

inline uint32_t get_work() {
    uint32_t ret;
    __asm__ volatile (".insn r %1, 1, 3, %0, x0, x0" : "=r"(ret) : "i"(RISCV_CUSTOM0));
    return ret;
}

template <uint32_t attr>
inline uint32_t get_attr() { 
    uint32_t ret;
    __asm__ volatile (".insn r %1, 2, 3, %0, x%[rs_id], x0" : "=r"(ret) : "i"(RISCV_CUSTOM0), [rs_id]"i"(attr));
    return ret;
}

template <uint32_t attr>
inline void set_attr(float val) { 
    __asm__ volatile (".insn r %0, 3, 3, x0, x%[rs_id], %[rs_val]" :: "i"(RISCV_CUSTOM0), [rs_id]"i"(attr), [rs_val] "f"(val));
}

template <uint32_t action>
inline void commit() {    
    __asm__ volatile (".insn r %[insn], 4, 3, x0, x%[rs_id], x0" :: [insn] "i" (RISCV_CUSTOM0), [rs_id]"i"(action));
}
} 
} 
