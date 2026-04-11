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

inline void traceRay(uint32_t payload_addr) {
    __asm__ volatile (".insn r %0, 0, 3, x0, %1, x0" :: "i"(RISCV_CUSTOM0), "r"(payload_addr));
}

inline int getWork() {
    int ret;
    __asm__ volatile (".insn r %1, 1, 3, %0, x0, x0" : "=r"(ret) : "i"(RISCV_CUSTOM0));
    return ret;
}

inline int getAttr(uint32_t rayID, uint32_t attrID) {
    int ret;
    __asm__ volatile (".insn r %1, 2, 3, %0, %2, %3" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(rayID), "r"(attrID));
    return ret;
}

inline void commit(uint32_t rayID, uint32_t actionID) {
    __asm__ volatile (".insn r %0, 3, 3, x0, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(rayID), "r"(actionID));
}

} 
} 
