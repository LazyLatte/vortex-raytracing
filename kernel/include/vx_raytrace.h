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
    uint32_t payload_addr
) {
    register uint32_t ret __asm__("x10");

    register float ox __asm__("f11") = ro_x;
    register float oy __asm__("f12") = ro_y;
    register float oz __asm__("f13") = ro_z;

    register float dx __asm__("f14") = rd_x;
    register float dy __asm__("f15") = rd_y;
    register float dz __asm__("f16") = rd_z;

    register float rs_tmin __asm__("f17") = tmin;
    register float rs_tmax __asm__("f18") = tmax;

    __asm__ volatile (
        ".insn r %[insn], 0, 3, %[rd_t], %[rs_ray], %[rs_payload_addr]"
        : [rd_t] "=r"(ret)
        : [insn] "i"(RISCV_CUSTOM0)
        , [rs_ray] "f"(ox), "f"(oy), "f"(oz), "f"(dx), "f"(dy), "f"(dz), "f"(rs_tmin), "f"(rs_tmax)
        , [rs_payload_addr] "r"(payload_addr)
    );
}

inline uint32_t get_work() {
    uint32_t ret;
    __asm__ volatile (".insn r %1, 1, 3, %0, x0, x0" : "=r"(ret) : "i"(RISCV_CUSTOM0));
    return ret;
}

// CHS / MS
template <uint32_t attr>
static __attribute__((always_inline)) uint32_t get_attr(uint32_t rayID) { 
    register uint32_t rs_rayID __asm__("x10") = rayID;
    
    uint32_t ret;
    __asm__ volatile (
        ".insn r %1, 2, 3, %0, x%[rs_id], x0" 
        : "=r"(ret) : "i"(RISCV_CUSTOM0)
        , [rs_id]"i"(attr), "r"(rs_rayID)
    );

    return ret;
}

// AHS / IS
template <uint32_t attr>
static __attribute__((always_inline)) uint32_t get_attr(uint32_t rayID, uint32_t hitID) {    
    register uint32_t rs_rayID __asm__("x10") = rayID;

    uint32_t ret;
    __asm__ volatile (
        ".insn r %1, 2, 3, %0, x%[rs_id], %[rs_hit]" 
        : "=r"(ret) : "i"(RISCV_CUSTOM0)
        , [rs_id]"i"(attr), "r"(rs_rayID)
        , [rs_hit] "r"(hitID)
    );

    return ret;
}

// AHS accept, AHS ignore, IS ignore
template <uint32_t action>
static __attribute__((always_inline)) void commit(uint32_t rayID, uint32_t hitID) {    
    register uint32_t rs_rayID __asm__("x10") = rayID;
    
    __asm__ volatile (
        ".insn r %[insn], 3, 3, x0, x%[rs_id], %[rs_hit]" 
        :: [insn] "i" (RISCV_CUSTOM0)
        , [rs_id]"i"(action), "r"(rs_rayID)
        , [rs_hit] "r"(hitID)
    );
}

// IS accept
static __attribute__((always_inline)) void commit(uint32_t rayID, uint32_t hitID, float t, float u, float v) {
    register uint32_t rs_rayID __asm__("x10") = rayID;
    register uint32_t rs_hitID __asm__("x11") = hitID;
    register float rs_t __asm__("f12") = t;
    register float rs_u __asm__("f13") = u;
    register float rs_v __asm__("f14") = v;
    
    __asm__ volatile (
        ".insn r %[insn], 4, 3, x0, %[rs_id], %[rs_hit]"
        :: [insn] "i"(RISCV_CUSTOM0)
        , [rs_id] "r"(rs_rayID) 
        , [rs_hit] "r"(rs_hitID), "f"(rs_t), "f"(rs_u), "f"(rs_v)
    );
}

inline void release_ray(uint32_t rayID) {
    __asm__ volatile (".insn r %0, 5, 3, x0, %1, x0" :: "i"(RISCV_CUSTOM0), "r"(rayID));
}

} 
} 
