#include "emulator.h"
#include "types.h"
#include <cstdint>
#include <iostream>
using namespace vortex;

static inline uint32_t bit(uint32_t x, int b) { return (x >> b) & 1u; }
static inline uint32_t bits(uint32_t x, int hi, int lo) { return (x >> lo) & ((1u << (hi - lo + 1)) - 1u); }
static inline uint32_t sext(uint32_t val, int width) {
    uint32_t m = 1u << (width - 1);
    return (val ^ m) - m;
}

// compressed register mapping: rd' (3 bits) -> x8..x15
static inline uint32_t rcp(uint32_t r3) { return 8u + r3; }

// Build I-type, S-type, U-type, R-type 32-bit encodings (RV32I)
static inline uint32_t ENCI(uint32_t imm12, uint32_t rs1, uint32_t funct3, uint32_t rd, uint32_t opcode) {
    return ((imm12 & 0xFFF) << 20) | ((rs1 & 31) << 15) | ((funct3 & 7) << 12) | ((rd & 31) << 7) | (opcode & 0x7F);
}
static inline uint32_t ENCR(uint32_t funct7, uint32_t rs2, uint32_t rs1, uint32_t funct3, uint32_t rd, uint32_t opcode) {
    return ((funct7 & 0x7F) << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | ((funct3 & 7) << 12) | ((rd & 31) << 7) | (opcode & 0x7F);
}
static inline uint32_t ENCS(uint32_t imm12, uint32_t rs2, uint32_t rs1, uint32_t funct3, uint32_t opcode) {
    uint32_t imm11_5 = (imm12 >> 5) & 0x7F;
    uint32_t imm4_0  = imm12 & 0x1F;
    return (imm11_5 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | ((funct3 & 7) << 12) | (imm4_0 << 7) | (opcode & 0x7F);
}
static inline uint32_t ENCU(uint32_t imm20, uint32_t rd, uint32_t opcode) {
    return ((imm20 & 0xFFFFF) << 12) | ((rd & 31) << 7) | (opcode & 0x7F);
}
static inline uint32_t ENCUJ(uint32_t imm21, uint32_t rd, uint32_t opcode) { // JAL
    // J-type bit shuffle: [20|10:1|11|19:12]
    uint32_t i = imm21 & 0x001FFFFF;
    uint32_t enc = ((i & (1<<20)) << 11)         // 20 -> 31
                 | ((i & 0x000007FE) << 20)      // 10:1 -> 30:21
                 | ((i & (1<<11)) << 9)          // 11 -> 20
                 | ((i & 0x000FF000));           // 19:12 -> 19:12 already aligned after shift
    return enc | ((rd & 31) << 7) | (opcode & 0x7F);
}
static inline uint32_t ENCB(uint32_t imm13, uint32_t rs2, uint32_t rs1, uint32_t funct3, uint32_t opcode) {
    // B-type: imm[12|10:5|4:1|11] -> [31|30:25|11:8|7]
    uint32_t i = imm13 & 0x1FFF;
    uint32_t imm12 = (i >> 12) & 1;
    uint32_t imm10_5 = (i >> 5) & 0x3F;
    uint32_t imm4_1 = (i >> 1) & 0xF;
    uint32_t imm11 = (i >> 11) & 1;
    uint32_t enc = (imm12 << 31) | (imm10_5 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) |
                   ((funct3 & 7) << 12) | (imm4_1 << 8) | (imm11 << 7) | (opcode & 0x7F);
    return enc;
}

// --- main ---------------------------------------------------------------

DecompResult Emulator::decompress(uint32_t word) {
    DecompResult out{};

    if ((word & 0x3) == 0x3) {
        out.instr32 = word;
        out.size    = 4;
        out.illegal = false;
        return out;
    }

    // 16-bit compressed
    const uint16_t h = static_cast<uint16_t>(word & 0xFFFF);
    const uint32_t quadrant = h & 0x3;              // 0,1,2
    const uint32_t funct3   = (h >> 13) & 0x7;

    out.size    = 2;
    out.illegal = false;

    switch (quadrant) {
    // ---------------- Quadrant 0 (opcode 00) ----------------
    case 0:
        switch (funct3) {
        case 0b000: { // C.ADDI4SPN -> ADDI rd', x2, nzimm
            uint32_t rd_ = rcp(bits(h, 4, 2));
            uint32_t nzuimm = (bits(h, 12, 11) << 4) | (bits(h,10, 7) << 6) | (bit(h, 5) << 3) | (bit(h, 6) << 2);
            if (nzuimm == 0) { out.illegal = true; break; }
            out.instr32 = ENCI(nzuimm, 2, 0b000, rd_, 0b0010011);
            break;
        }
        case 0b010: { // C.LW -> LW rd', offset(rs1')
            uint32_t rd_  = rcp(bits(h, 4, 2));
            uint32_t rs1_ = rcp(bits(h, 9, 7));
            uint32_t uimm = (bit(h, 5) << 6) | (bits(h, 12, 10) << 3) | (bit(h, 6) << 2);
            out.instr32 = ENCI(uimm, rs1_, 0b010, rd_, 0b0000011);
            break;
        }
        
        case 0b011: { // C.FLW -> FLW rd', offset(rs1')
            uint32_t rd_  = rcp(bits(h, 4, 2));
            uint32_t rs1_ = rcp(bits(h, 9, 7));
            uint32_t uimm = (bit(h, 5) << 6) | (bits(h, 12, 10) << 3) | (bit(h, 6) << 2);
            out.instr32 = ENCI(uimm, rs1_, 0b010, rd_, 0b0000111);
            break;
        }
        case 0b110: { // C.SW -> SW rs2', offset(rs1')
            uint32_t rs2_ = rcp(bits(h, 4, 2));
            uint32_t rs1_ = rcp(bits(h, 9, 7));
            uint32_t uimm = (bit(h, 5) << 6) | (bits(h, 12, 10) << 3) | (bit(h, 6) << 2);
            out.instr32 = ENCS(uimm, rs2_, rs1_, 0b010, 0b0100011);
            break;
        }
        
        case 0b111: { // C.FSW
            uint32_t rs2_ = rcp(bits(h, 4, 2));
            uint32_t rs1_ = rcp(bits(h, 9, 7));
            uint32_t uimm = (bit(h, 5) << 6) | (bits(h, 12, 10) << 3) | (bit(h, 6) << 2);
            out.instr32 = ENCS(uimm, rs2_, rs1_, 0b010, 0b0100111);
            break;
        }
        default:
            out.illegal = true; break;
        }
        break;

    // ---------------- Quadrant 1 (opcode 01) ----------------
    case 1:
        switch (funct3) {
        case 0b000: { // C.ADDI -> ADDI rd, rd, imm
            uint32_t rd = bits(h, 11, 7);
            int32_t imm = static_cast<int32_t>(sext(((bit(h,12)<<5) | bits(h,6,2)), 6));
            if (rd == 0) { out.illegal = true; break; } // C.NOP is ADDI x0,x0,0 (legal), handle separately
            out.instr32 = ENCI(imm & 0xFFF, rd, 0b000, rd, 0b0010011);
            // Special-case C.NOP:
            if (rd == 0 && (imm == 0)) {
                out.illegal = false;
                out.instr32 = ENCI(0, 0, 0b000, 0, 0b0010011); // ADDI x0,x0,0
            }
            break;
        }
        case 0b001:{
            uint32_t imm =
                (bit(h,12)<<11) | (bit(h,8)<<10) | (bits(h,10,9)<<8) | (bit(h,6)<<7) |
                (bit(h,7)<<6) | (bit(h,2)<<5) | (bit(h,11)<<4) | (bits(h,5,3)<<1);
            int32_t simm = static_cast<int32_t>(sext(imm, 12));
            uint32_t imm21 = static_cast<uint32_t>(simm) & 0x001FFFFF;
            out.instr32 = ENCUJ(imm21, 1, 0b1101111);
            break;
        }
        case 0b010: { // C.LI -> ADDI rd, x0, imm
            uint32_t rd = bits(h, 11, 7);
            int32_t imm = static_cast<int32_t>(sext(((bit(h,12)<<5) | bits(h,6,2)), 6));
            if (rd == 0) { out.illegal = true; break; }
            out.instr32 = ENCI(imm & 0xFFF, 0, 0b000, rd, 0b0010011);
            break;
        }
        case 0b011: {
            uint32_t rd = bits(h, 11, 7);
            if (rd == 2) {
                // C.ADDI16SP -> ADDI x2, x2, imm
                // imm: [9|4|6|8:7|5] -> bits from h: 12|6|5|4:3|2
                int32_t imm = (bit(h,12)<<9) | (bit(h,6)<<4) | (bit(h,5)<<6) | (bits(h,4,3)<<7) | (bit(h,2)<<5);
                imm = static_cast<int32_t>(sext(imm, 10));
                if (imm == 0) { out.illegal = true; break; }
                out.instr32 = ENCI(imm & 0xFFF, 2, 0b000, 2, 0b0010011);
            } else {
                // C.LUI -> LUI rd, imm (rd!=x0,x2)
                int32_t imm = static_cast<int32_t>(sext((bit(h,12)<<17) | (bits(h,6,2)<<12), 18));
                if (rd == 0 || rd == 2 || imm == 0) { out.illegal = true; break; }
                out.instr32 = ENCU((imm >> 12), rd, 0b0110111);
            }
            break;
        }
        case 0b100: {
            uint32_t subfunct = bits(h, 11, 10);
            if (subfunct == 0b00) { // C.SRLI
                uint32_t rd_ = rcp(bits(h,9,7));
                uint32_t sh  = (bit(h,12)<<5) | bits(h,6,2);
                out.instr32 = ENCI(sh, rd_, 0b101, rd_, 0b0010011);
            } else if (subfunct == 0b01) { // C.SRAI
                uint32_t rd_ = rcp(bits(h,9,7));
                uint32_t sh  = (bit(h,12)<<5) | bits(h,6,2);
                out.instr32 = ENCI(sh, rd_, 0b101, rd_, 0b0010011) | (0x40000000u); // add funct7=0100000 via bit 30
            } else if (subfunct == 0b10) { // C.ANDI
                uint32_t rd_ = rcp(bits(h,9,7));
                int32_t imm = static_cast<int32_t>(sext(((bit(h,12)<<5) | bits(h,6,2)), 6));
                out.instr32 = ENCI(imm & 0xFFF, rd_, 0b111, rd_, 0b0010011);
            } else { // 0b11: C.SUB/XOR/OR/AND (register form)
                uint32_t rd_  = rcp(bits(h,9,7));
                uint32_t rs2_ = rcp(bits(h,4,2));
                uint32_t op2  = bits(h, 6,5);
                //check if bit 12 == 0
                switch (op2) {
                    case 0b00: // C.SUB -> SUB rd', rd', rs2'
                        out.instr32 = ENCR(0b0100000, rs2_, rd_, 0b000, rd_, 0b0110011); break;
                    case 0b01: // C.XOR
                        out.instr32 = ENCR(0b0000000, rs2_, rd_, 0b100, rd_, 0b0110011); break;
                    case 0b10: // C.OR
                        out.instr32 = ENCR(0b0000000, rs2_, rd_, 0b110, rd_, 0b0110011); break;
                    case 0b11: // C.AND
                        out.instr32 = ENCR(0b0000000, rs2_, rd_, 0b111, rd_, 0b0110011); break;
                }
            }
            break;
        }
        case 0b101: { // C.J -> JAL x0, imm
            uint32_t imm =
                (bit(h,12)<<11) | (bit(h,8)<<10) | (bits(h,10,9)<<8) | (bit(h,6)<<7) |
                (bit(h,7)<<6) | (bit(h,2)<<5) | (bit(h,11)<<4) | (bits(h,5,3)<<1);
            int32_t simm = static_cast<int32_t>(sext(imm, 12));
            uint32_t imm21 = static_cast<uint32_t>(simm) & 0x001FFFFF;
            out.instr32 = ENCUJ(imm21, 0, 0b1101111);
            break;
        }
        case 0b110: { // C.BEQZ -> BEQ rs1', x0, imm
            uint32_t rs1_ = rcp(bits(h, 9,7));
            uint32_t imm = (bit(h,12)<<8) | (bit(h,6)<<7) | (bit(h,5)<<6) | (bit(h,2)<<5) | (bits(h,11,10)<<3) | (bits(h,4,3)<<1);
            int32_t simm = static_cast<int32_t>(sext(imm, 9));
            uint32_t imm13 = static_cast<uint32_t>(simm) & 0x1FFF;
            out.instr32 = ENCB(imm13, 0, rs1_, 0b000, 0b1100011);
            break;
        }
        case 0b111: { // C.BNEZ -> BNE rs1', x0, imm
            uint32_t rs1_ = rcp(bits(h, 9,7));
            uint32_t imm = (bit(h,12)<<8) | (bit(h,6)<<7) | (bit(h,5)<<6) | (bit(h,2)<<5) | (bits(h,11,10)<<3) | (bits(h,4,3)<<1);
            int32_t simm = static_cast<int32_t>(sext(imm, 9));
            uint32_t imm13 = static_cast<uint32_t>(simm) & 0x1FFF;
            out.instr32 = ENCB(imm13, 0, rs1_, 0b001, 0b1100011);
            break;
        }
        default:
            out.illegal = true; break;
        }
        break;

    // ---------------- Quadrant 2 (opcode 10) ----------------
    case 2:
        switch (funct3) {
        case 0b000: { // C.SLLI -> SLLI rd, rd, shamt
            uint32_t rd = bits(h, 11, 7);
            uint32_t sh = (bit(h,12)<<5) | bits(h,6,2);
            if (rd == 0) {
                out.illegal = true;
                break;
            }
            out.instr32 = ENCI(sh, rd, 0b001, rd, 0b0010011);
            break;
        }

        case 0b100: {
            uint32_t rd  = bits(h, 11, 7);
            uint32_t rs2 = bits(h, 6, 2);
            uint32_t s12 = bit(h, 12);

            // rs2 == 0 → JR / JALR / EBREAK or illegal
            if (rs2 == 0) {
                if (s12 == 0) {
                    // C.JR
                    // rd != 0, rs2 = 0
                    if (rd == 0) {
                        out.illegal = true;  // reserved encoding
                        break;
                    }
                    // JALR x0, 0(rd)
                    out.instr32 = ENCI(0, rd, 0b000, 0, 0b1100111);
                } else {
                    // s12 == 1, rs2 == 0
                    if (rd == 0) {
                        // C.EBREAK -> EBREAK (SYSTEM, imm=1)
                        out.instr32 = 0x00100073;
                    } else {
                        // C.JALR
                        // JALR x1, 0(rd)
                        out.instr32 = ENCI(0, rd, 0b000, 1, 0b1100111);
                    }
                }
            } else {
                // rs2 != 0 → MV / ADD
                if (rd == 0) {
                    out.illegal = true; // rd must not be x0
                    break;
                }
                if (s12 == 0) {
                    // C.MV -> ADD rd, x0, rs2
                    out.instr32 = ENCR(0b0000000, rs2, 0, 0b000, rd, 0b0110011);
                } else {
                    // C.ADD -> ADD rd, rd, rs2
                    out.instr32 = ENCR(0b0000000, rs2, rd, 0b000, rd, 0b0110011);
                }
            }
            break;
        }
        case 0b110: { // C.SWSP -> SW rs2, offset[7:2](x2)
            uint32_t rs2 = bits(h, 6, 2);
            uint32_t imm = (bits(h, 12, 9) << 2) | (bits(h, 8, 7)  << 6);  
            out.instr32 = ENCS(imm, rs2, 2, 0b010, 0b0100011);
            break;
        }
        default:
            out.illegal = true; break;
        }
        break;

    default:
        out.illegal = true; break;
    }
    if(out.illegal){
        std::cout << "Illegal 16-bit! Quadrant: " << quadrant << ", func3: " << funct3 << std::endl;
    }
    return out;
}
