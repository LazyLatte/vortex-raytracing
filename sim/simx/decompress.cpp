#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <util.h>
#include "debug.h"
#include "types.h"
#include "emulator.h"
#include "arch.h"
#include "instr.h"

// ----- helpers -----
static inline uint32_t bit(uint32_t x, int b) { return (x >> b) & 1u; }
static inline uint32_t bits(uint32_t x, int hi, int lo) { return (x >> lo) & ((1u << (hi - lo + 1)) - 1u); }
static inline uint32_t sext(uint32_t v, int w) { uint32_t m = 1u << (w - 1); return (v ^ m) - m; }
static inline uint32_t rcp(uint32_t r3) { return 8u + r3; } // compressed reg x8..x15

// encoders (RV32I)
static inline uint32_t ENCI(uint32_t imm12, uint32_t rs1, uint32_t f3, uint32_t rd, uint32_t opc) {
    return ((imm12 & 0xFFF) << 20) | ((rs1 & 31) << 15) | ((f3 & 7) << 12) | ((rd & 31) << 7) | (opc & 0x7F);
}
static inline uint32_t ENCR(uint32_t f7, uint32_t rs2, uint32_t rs1, uint32_t f3, uint32_t rd, uint32_t opc) {
    return ((f7 & 0x7F) << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | ((f3 & 7) << 12) | ((rd & 31) << 7) | (opc & 0x7F);
}
static inline uint32_t ENCS(uint32_t imm12, uint32_t rs2, uint32_t rs1, uint32_t f3, uint32_t opc) {
    uint32_t i11_5 = (imm12 >> 5) & 0x7F, i4_0 = imm12 & 0x1F;
    return (i11_5 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | ((f3 & 7) << 12) | (i4_0 << 7) | (opc & 0x7F);
}
static inline uint32_t ENCU(uint32_t imm20, uint32_t rd, uint32_t opc) {
    return ((imm20 & 0xFFFFF) << 12) | ((rd & 31) << 7) | (opc & 0x7F);
}
static inline uint32_t ENCUJ(int32_t imm21, uint32_t rd, uint32_t opc) { // JAL
    // imm21 is signed, bit 0 must be 0 (J alignment). We shuffle J-type: [20|10:1|11|19:12]
    uint32_t i = (uint32_t)imm21 & 0x001FFFFF;
    uint32_t enc = ((i & (1<<20)) << 11) | ((i & 0x000007FE) << 20) | ((i & (1<<11)) << 9) | ((i & 0x000FF000));
    return enc | ((rd & 31) << 7) | (opc & 0x7F);
}
static inline uint32_t ENCB(int32_t imm13, uint32_t rs2, uint32_t rs1, uint32_t f3, uint32_t opc) {
    uint32_t i = (uint32_t)imm13 & 0x1FFF;                  // [12:0], bit0 must be 0 (B alignment)
    uint32_t i12 = (i >> 12) & 1, i10_5 = (i >> 5) & 0x3F, i4_1 = (i >> 1) & 0xF, i11 = (i >> 11) & 1;
    return (i12 << 31) | (i10_5 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) |
           ((f3 & 7) << 12) | (i4_1 << 8) | (i11 << 7) | (opc & 0x7F);
}

// ----- main -----
uint32_t expand_compressed(uint16_t h, bool &illegal /*out*/) {
    illegal = false;

    const uint32_t quadrant = h & 0x3;      // 0,1,2 (3 would be 32-bit)
    const uint32_t f3       = (h >> 13) & 7;

    switch (quadrant) {

// ================= Quadrant 0 (opcode 00) =================
    case 0: switch (f3) {
        case 0b000: { // C.ADDI4SPN -> ADDI rd', x2, nzimm
            uint32_t rd_   = rcp(bits(h, 4, 2));
            uint32_t nzimm = (bit(h,12)<<5) | (bits(h,6,5)<<3) | (bits(h,11,7)<<6) | (bit(h,2)<<2);
            if (nzimm == 0) { illegal = true; break; }
            return ENCI(nzimm, 2, 0b000, rd_, 0b0010011); // ADDI
        }
        case 0b010: { // C.LW -> LW rd', offset(rs1')
            uint32_t rd_  = rcp(bits(h,4,2));
            uint32_t rs1_ = rcp(bits(h,9,7));
            uint32_t uimm = (bit(h,5)<<6) | (bit(h,12)<<5) | (bits(h,11,10)<<3) | (bit(h,6)<<2);
            return ENCI(uimm, rs1_, 0b010, rd_, 0b0000011); // LW
        }
        case 0b110: { // C.SW -> SW rs2', offset(rs1')
            uint32_t rs2_ = rcp(bits(h,4,2));
            uint32_t rs1_ = rcp(bits(h,9,7));
            uint32_t uimm = (bit(h,5)<<6) | (bit(h,12)<<5) | (bits(h,11,10)<<3) | (bit(h,6)<<2);
            return ENCS(uimm, rs2_, rs1_, 0b010, 0b0100011); // SW
        }
        default: illegal = true; break;
    } break;

// ================= Quadrant 1 (opcode 01) =================
    case 1: switch (f3) {
        case 0b000: { // C.NOP / C.ADDI -> ADDI rd, rd, imm
            uint32_t rd  = bits(h,11,7);
            int32_t  imm = (int32_t)sext((bit(h,12)<<5) | bits(h,6,2), 6);
            if (rd == 0) { // rd==x0 -> C.NOP only legal if imm==0
                if (imm == 0) return ENCI(0, 0, 0b000, 0, 0b0010011); // ADDI x0,x0,0
                illegal = true; break;
            }
            return ENCI((uint32_t)(imm & 0xFFF), rd, 0b000, rd, 0b0010011); // ADDI
        }
        case 0b001: { // C.JAL (RV32) -> JAL x1, imm
            // imm: [11|4|9:8|10|6|7|3:1|5] << 1
            uint32_t i = (bit(h,12)<<11) | (bit(h,8)<<10) | (bits(h,10,9)<<8) | (bit(h,6)<<7) |
                         (bit(h,7)<<6)  | (bits(h,2,1)<<3) | (bit(h,11)<<2) | (bit(h,5)<<1);
            int32_t simm = (int32_t)sext(i << 1, 12+1);
            return ENCUJ(simm, 1, 0b1101111); // JAL x1, simm
        }
        case 0b010: { // C.LI -> ADDI rd, x0, imm
            uint32_t rd  = bits(h,11,7);
            int32_t  imm = (int32_t)sext((bit(h,12)<<5) | bits(h,6,2), 6);
            if (rd == 0) { illegal = true; break; }
            return ENCI((uint32_t)(imm & 0xFFF), 0, 0b000, rd, 0b0010011);
        }
        case 0b011: { // C.ADDI16SP or C.LUI
            uint32_t rd = bits(h,11,7);
            if (rd == 2) { // C.ADDI16SP
                // imm: [9|4|6|8:7|5] from bits 12,4,6,3:2,5 -> then <<0 (I-type)
                uint32_t raw = (bit(h,12)<<9) | (bit(h,4)<<4) | (bit(h,6)<<6) | (bits(h,3,2)<<7) | (bit(h,5)<<5);
                int32_t imm  = (int32_t)sext(raw, 10);
                if (imm == 0) { illegal = true; break; }
                return ENCI((uint32_t)(imm & 0xFFF), 2, 0b000, 2, 0b0010011); // ADDI x2,x2,imm
            } else { // C.LUI -> LUI rd, imm (rd!=x0,x2; imm!=0)
                int32_t imm = (int32_t)sext(((bit(h,12)<<17) | (bits(h,6,2)<<12)), 18);
                if (rd == 0 || rd == 2 || imm == 0) { illegal = true; break; }
                return ENCU((uint32_t)((imm >> 12) & 0xFFFFF), rd, 0b0110111);
            }
        }
        case 0b101: { // C.J -> JAL x0, imm
            uint32_t i = (bit(h,12)<<11) | (bit(h,8)<<10) | (bits(h,10,9)<<8) | (bit(h,6)<<7) |
                         (bit(h,7)<<6)  | (bits(h,2,1)<<3) | (bit(h,11)<<2) | (bit(h,5)<<1);
            int32_t simm = (int32_t)sext(i << 1, 12+1);
            return ENCUJ(simm, 0, 0b1101111); // JAL x0, simm
        }
        case 0b110: { // C.BEQZ -> BEQ rs1', x0, imm
            uint32_t rs1_ = rcp(bits(h,9,7));
            uint32_t i = (bit(h,12)<<8) | (bit(h,6)<<7) | (bit(h,5)<<6) | (bit(h,2)<<5) | (bits(h,11,10)<<3) | (bit(h,3)<<1);
            int32_t simm = (int32_t)sext(i << 1, 9+1);
            return ENCB(simm, 0, rs1_, 0b000, 0b1100011); // BEQ
        }
        case 0b111: { // C.BNEZ -> BNE rs1', x0, imm
            uint32_t rs1_ = rcp(bits(h,9,7));
            uint32_t i = (bit(h,12)<<8) | (bit(h,6)<<7) | (bit(h,5)<<6) | (bit(h,2)<<5) | (bits(h,11,10)<<3) | (bit(h,3)<<1);
            int32_t simm = (int32_t)sext(i << 1, 9+1);
            return ENCB(simm, 0, rs1_, 0b001, 0b1100011); // BNE
        }
        default: illegal = true; break;
    } break;

// ================= Quadrant 2 (opcode 10) =================
    case 2: switch (f3) {
        case 0b000: { // C.SLLI -> SLLI rd, rd, shamt
            uint32_t rd = bits(h,11,7);
            uint32_t sh = (bit(h,12)<<5) | bits(h,6,2);
            if (rd == 0) { illegal = true; break; } // rd!=x0 in RV32C
            return ENCI(sh, rd, 0b001, rd, 0b0010011); // SLLI
        }
        case 0b010: { // C.LWSP -> LW rd, uimm(x2)
            uint32_t rd   = bits(h,11,7);
            uint32_t uimm = (bit(h,12)<<5) | (bits(h,4,2)<<2) | (bits(h,6,6)<<6); // [5|4:2|6] -> <<0 with I-type
            if (rd == 0) { illegal = true; break; }
            return ENCI(uimm, 2, 0b010, rd, 0b0000011);
        }
        case 0b100: {
            uint32_t rs2 = bits(h,6,2);
            uint32_t rd  = bits(h,11,7);
            if (rs2 == 0) {
                // JR / JALR / MV/ADD with rs2==0 patterns
                if (bit(h,12) == 0 && rd != 0) { // C.JR -> JALR x0, 0(rd)
                    return ENCI(0, rd, 0b000, 0, 0b1100111);
                }
                if (bit(h,12) == 1 && rd != 0) { // C.JALR -> JALR x1, 0(rd)
                    return ENCI(0, rd, 0b000, 1, 0b1100111);
                }
                illegal = true; break; // rd==0 reserved
            } else {
                if (bit(h,12) == 0) { // C.MV -> ADD rd, x0, rs2
                    if (rd == 0) { illegal = true; break; }
                    return ENCR(0, rs2, 0, 0b000, rd, 0b0110011); // ADD rd,x0,rs2
                } else { // C.ADD -> ADD rd, rd, rs2
                    if (rd == 0) { illegal = true; break; }
                    return ENCR(0, rs2, rd, 0b000, rd, 0b0110011);
                }
            }
            break;
        }
        case 0b110: { // C.SWSP -> SW rs2, uimm(x2)
            uint32_t rs2  = bits(h,6,2);
            uint32_t uimm = (bits(h,8,7)<<6) | (bits(h,12,9)<<2); // [7:6|12:9] mapped to S-type [11:5|4:0]
            return ENCS(uimm, rs2, 2, 0b010, 0b0100011);
        }
        case 0b101: // (unused in RV32C)
        case 0b011: // (RV64/128 variants; ignore for RV32)
        case 0b001: // SRLI/SRAI/ANDI live in f3=100 with subfunct; handled below? No—those are here too:
        default: {
            // Handle f3=100 compressed-aligned ops: SRLI/SRAI/ANDI and SUB/XOR/OR/AND (with compact regs)
            if (f3 == 0b100) {
                if (bits(h,11,10) <= 0b01) { // SRLI/SRAI
                    uint32_t rd_ = rcp(bits(h,9,7));
                    uint32_t sh  = (bit(h,12)<<5) | bits(h,6,2);
                    if (bits(h,11,10) == 0b00) { // C.SRLI
                        return ENCI(sh, rd_, 0b101, rd_, 0b0010011); // SRLI
                    } else { // C.SRAI
                        // SRAI enc: same I-format with funct7=0100000 indicated by bit 30
                        return ENCI(sh, rd_, 0b101, rd_, 0b0010011) | (1u << 30);
                    }
                } else if (bits(h,11,10) == 0b10) { // C.ANDI
                    uint32_t rd_ = rcp(bits(h,9,7));
                    int32_t  imm = (int32_t)sext((bit(h,12)<<5) | bits(h,6,2), 6);
                    return ENCI((uint32_t)(imm & 0xFFF), rd_, 0b111, rd_, 0b0010011);
                } else { // 0b11 -> C.SUB/XOR/OR/AND
                    uint32_t rd_  = rcp(bits(h,9,7));
                    uint32_t rs2_ = rcp(bits(h,4,2));
                    switch (bits(h,6,5)) {
                        case 0b00: return ENCR(0b0100000, rs2_, rd_, 0b000, rd_, 0b0110011); // SUB
                        case 0b01: return ENCR(0b0000000, rs2_, rd_, 0b100, rd_, 0b0110011); // XOR
                        case 0b10: return ENCR(0b0000000, rs2_, rd_, 0b110, rd_, 0b0110011); // OR
                        case 0b11: return ENCR(0b0000000, rs2_, rd_, 0b111, rd_, 0b0110011); // AND
                    }
                }
            }
            illegal = true; break;
        }}
    break;

    default:
        illegal = true; break;
    }

    // Fallback
    return 0;
}


uint32_t Emulator::decompress(uint32_t instr_word){
  uint16_t half = instr_word & 0xFFFF;
  if ((half & 0x3) == 0x3) {
    return instr_word;
  } else {
    bool illegal = false;
    uint32_t exp = expand_compressed(half, illegal);
    if (illegal) {
        // raise trap / return encoding for ILLEGAL
        // e.g., use 0x00000013 (ADDI x0,x0,0) as a safe NOP *or* signal fault per your sim policy
    }
    return exp;
  }
}