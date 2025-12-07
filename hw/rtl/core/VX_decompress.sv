// VX_decompress.sv
`include "VX_define.vh"

module VX_decompress import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input  wire                     clk,
    input  wire                     reset,

    // Input side (from icache + tag_store)
    input  wire                     valid_in,
    output wire                     ready_in,
    input  wire [PC_BITS-1:0]       PC_in,
    input  wire [`NUM_THREADS-1:0]  tmask_in,
    input  wire [NW_WIDTH-1:0]      wid_in,
    input  wire [UUID_WIDTH-1:0]    uuid_in,
    input  wire [1:0]               state_in,
    input  wire [31:0]              word_in, // full icache line/word

    // Output side (to fetch_if)
    output reg                      valid_out,
    input  wire                     ready_out,
    output reg  [PC_BITS-1:0]       PC_out,
    output reg  [`NUM_THREADS-1:0]  tmask_out,
    output reg  [NW_WIDTH-1:0]      wid_out,
    output reg  [UUID_WIDTH-1:0]    uuid_out,
    output reg  [1:0]               state_out,
    output reg  [31:0]              instr_out,

    output reg                      rvc_out,
    output reg                      incomplete_out
);
    `UNUSED_SPARAM (INSTANCE_ID)

    //-------------------------------------------------------------------------
    // Local state
    //-------------------------------------------------------------------------

    // Hold the current word and metadata when accepted
    reg [31:0]                    word_reg;
    reg [PC_BITS-1:0]             pc_reg;
    reg [`NUM_THREADS-1:0]        tmask_reg;
    reg [NW_WIDTH-1:0]            wid_reg;
    reg [UUID_WIDTH-1:0]          uuid_reg;
    reg [1:0]                     state_reg;
    reg                           have_word;

    reg [`NUM_WARPS-1:0][15:0]    half_reg;

    function automatic logic is_rvc (input [1:0] hw);
        return hw != 2'b11;
    endfunction

    // --- RV32I/F opcode constants used by decompressor ---
    `define INST_L      7'b0000011  // LOAD
    `define INST_S      7'b0100011  // STORE
    `define INST_B      7'b1100011  // BRANCH
    `define INST_JAL    7'b1101111  // JAL
    `define INST_JALR   7'b1100111  // JALR
    `define INST_LUI    7'b0110111  // LUI
    `define INST_I      7'b0010011  // OP-IMM
    `define INST_R      7'b0110011  // OP
    `define INST_FL     7'b0000111  // LOAD-FP (flw)
    `define INST_FS     7'b0100111  // STORE-FP (fsw)

`ifdef XLEN_64
    `define INST_I_W    7'b0011011  // OP-IMM-32
    `define INST_R_W    7'b0111011  // OP-32
`endif

    function automatic [31:0] decompress16 (input [15:0] instr_i);
        logic [2:0] func3;
        logic [4:0] rd, rs1, rs2;
        logic [4:0] rdp, rs1p, rs2p;

        logic [11:0] lsw_imm, lwsp_imm, swsp_imm;
    `ifdef XLEN_64
        logic [11:0] lsd_imm, ldsp_imm, sdsp_imm;
    `endif
        logic [11:0] i_imm, b_imm, x_imm, w_imm;
        logic [19:0] j_imm;

        logic [31:0] instr_o;

        func3 = instr_i[15:13];
        rd    = instr_i[11:7];
        rs1   = rd;
        rs2   = instr_i[6:2];

        // compact regs x8..x15
        rdp   = {2'b01, instr_i[4:2]};
        rs1p  = {2'b01, instr_i[9:7]};
        rs2p  = rdp;

        // immediates
        lsw_imm  = {5'b0, instr_i[5], instr_i[12:10], instr_i[6], 2'b0};
        lwsp_imm = {4'b0, instr_i[3:2], instr_i[12], instr_i[6:4], 2'b0};
        swsp_imm = {4'b0, instr_i[8:7], instr_i[12:9], 2'b0};

    `ifdef XLEN_64
        lsd_imm  = {4'b0, instr_i[6:5], instr_i[12:10], 3'b0};
        ldsp_imm = {3'b0, instr_i[4:2], instr_i[12], instr_i[6:5], 3'b0};
        sdsp_imm = {3'b0, instr_i[9:7], instr_i[12:10], 3'b0};
    `endif

        i_imm  = {{7{instr_i[12]}}, instr_i[6:2]};
        b_imm  = {{5{instr_i[12]}}, instr_i[6:5], instr_i[2], instr_i[11:10], instr_i[4:3]};
        j_imm  = {{10{instr_i[12]}}, instr_i[8], instr_i[10:9], instr_i[6], instr_i[7],
                  instr_i[2], instr_i[11], instr_i[5:3]};
        x_imm  = {{3{instr_i[12]}}, instr_i[4:3], instr_i[5], instr_i[2], instr_i[6], 4'b0};
        w_imm  = {2'd0, instr_i[10:7], instr_i[12:11], instr_i[5], instr_i[6], 2'b0};

        instr_o = '0;

        case (instr_i[1:0])

        // ------------------------
        // Quadrant 0
        // ------------------------
        2'b00: begin
            case (func3)
            3'b000: // c.addi4spn -> addi rd', x2, imm
                instr_o = {w_imm, 5'd2, 3'b000, rdp, `INST_I};

        `ifdef FLEN_64
            3'b001: // c.fld -> fld rd', rs1', imm
                instr_o = {lsd_imm, rs1p, 3'b011, rdp, `INST_FL};
        `endif

            3'b010: // c.lw -> lw rd', rs1', imm
                instr_o = {lsw_imm, rs1p, 3'b010, rdp, `INST_L};

        `ifdef XLEN_64
            3'b011: // c.ld -> ld rd', rs1', imm
                instr_o = {lsd_imm, rs1p, 3'b011, rdp, `INST_L};
        `else
            3'b011: // c.flw -> flw rd', rs1', imm
                instr_o = {lsw_imm, rs1p, 3'b010, rdp, `INST_FL};
        `endif

        `ifdef FLEN_64
            3'b101: // c.fsd -> fsd rs2', rs1', imm
                instr_o = {lsd_imm[11:5], rs2p, rs1p, 3'b011, lsd_imm[4:0], `INST_FS};
        `endif

            3'b110: // c.sw -> sw rs2', rs1', imm
                instr_o = {lsw_imm[11:5], rs2p, rs1p, 3'b010, lsw_imm[4:0], `INST_S};

        `ifdef XLEN_64
            3'b111: // c.sd -> sd rs2', rs1', imm
                instr_o = {lsd_imm[11:5], rs2p, rs1p, 3'b011, lsd_imm[4:0], `INST_S};
        `else
            3'b111: // c.fsw -> fsw rs2', rs1', imm
                instr_o = {lsw_imm[11:5], rs2p, rs1p, 3'b010, lsw_imm[4:0], `INST_FS};
        `endif

            default:
                instr_o = 32'h00000013; // NOP
            endcase
        end

        // ------------------------
        // Quadrant 1
        // ------------------------
        2'b01: begin
            case (func3)
            3'b000: begin
                if ((rd == 5'd0) && (i_imm == 12'd0)) begin
                    instr_o = 32'h00000013; // C.NOP
                end else begin
                    instr_o = {i_imm, rd, 3'b000, rd, `INST_I}; // C.ADDI
                end
            end
        `ifdef XLEN_64
            3'b001: // c.addiw -> addiw rd, rd, imm
                instr_o = {i_imm, rd, 3'b000, rd, `INST_I_W};
        `endif

            3'b010: // c.li -> addi rd, x0, imm
                instr_o = {i_imm, 5'd0, 3'b000, rd, `INST_I};

            3'b011: begin
                if (rd == 5'd2) begin
                    // c.addi16sp -> addi x2, x2, imm
                    instr_o = {x_imm, 5'd2, 3'b000, 5'd2, `INST_I};
                end else begin
                    // c.lui -> lui rd, imm
                    instr_o = {{{8{i_imm[11]}}, i_imm}, rd, `INST_LUI};
                end
            end

            3'b100: begin
                case (instr_i[11:10])
                2'b00: // c.srli
                    instr_o = {{6'b000000, i_imm[5:0]}, rs1p, 3'b101, rs1p, `INST_I};
                2'b01: // c.srai
                    instr_o = {{6'b010000, i_imm[5:0]}, rs1p, 3'b101, rs1p, `INST_I};
                2'b10: // c.andi
                    instr_o = {i_imm, rs1p, 3'b111, rs1p, `INST_I};
                2'b11: begin
                    case ({instr_i[12], instr_i[6:5]})
                    3'b000: // c.sub
                        instr_o = {7'b0100000, rs2p, rs1p, 3'b000, rs1p, `INST_R};
                    3'b001: // c.xor
                        instr_o = {7'b0000000, rs2p, rs1p, 3'b100, rs1p, `INST_R};
                    3'b010: // c.or
                        instr_o = {7'b0000000, rs2p, rs1p, 3'b110, rs1p, `INST_R};
                    3'b011: // c.and
                        instr_o = {7'b0000000, rs2p, rs1p, 3'b111, rs1p, `INST_R};
        `ifdef XLEN_64
                    3'b100: // c.subw
                        instr_o = {7'b0100000, rs2p, rs1p, 3'b000, rs1p, `INST_R_W};
                    3'b101: // c.addw
                        instr_o = {7'b0000000, rs2p, rs1p, 3'b000, rs1p, `INST_R_W};
        `endif
                    default:
                        instr_o = 32'h00000013;
                    endcase
                end
                default:
                    instr_o = 32'h00000013;
                endcase
            end

            3'b101: // c.j
                instr_o = {j_imm[19], j_imm[9:0], j_imm[10], j_imm[18:11], 5'd0, `INST_JAL};

            3'b110: // c.beqz
                instr_o = {b_imm[11], b_imm[9:4], 5'd0, rs1p, 3'b000, b_imm[3:0], b_imm[10], `INST_B};

            3'b111: // c.bnez
                instr_o = {b_imm[11], b_imm[9:4], 5'd0, rs1p, 3'b001, b_imm[3:0], b_imm[10], `INST_B};

            default:
                instr_o = 32'h00000013;
            endcase
        end

        // ------------------------
        // Quadrant 2
        // ------------------------
        2'b10: begin
            case (func3)
            3'b000:  // c.slli
                instr_o = {{6'b000000, i_imm[5:0]}, rd, 3'b001, rd, `INST_I};

        `ifdef FLEN_64
            3'b001: // c.fldsp
                instr_o = {ldsp_imm, 5'd2, 3'b011, rd, `INST_FL};
        `endif

            3'b010: // c.lwsp
                instr_o = {lwsp_imm, 5'd2, 3'b010, rd, `INST_L};

        `ifdef XLEN_64
            3'b011: // c.ldsp
                instr_o = {ldsp_imm, 5'd2, 3'b011, rd, `INST_L};
        `else
            3'b011: // c.flwsp
                instr_o = {lwsp_imm, 5'd2, 3'b010, rd, `INST_FL};
        `endif

            3'b100: begin
                if (instr_i[12] == 1'b0) begin
                    if (rs2 == 5'd0) begin
                        // c.jr
                        instr_o = {12'd0, rs1, 3'b000, 5'd0, `INST_JALR};
                    end else begin
                        // c.mv
                        instr_o = {7'b0000000, rs2, 5'd0, 3'b000, rd, `INST_R};
                    end
                end else begin
                    if (rs2 == 5'd0) begin
                        if (rs1 == 5'd0) begin
                            // c.ebreak
                            instr_o = 32'b000000000001_00000_000_00000_1110011;
                        end else begin
                            // c.jalr
                            instr_o = {12'd0, rs1, 3'b000, 5'd1, `INST_JALR};
                        end
                    end else begin
                        // c.add
                        instr_o = {7'b0000000, rs2, rd, 3'b000, rd, `INST_R};
                    end
                end
            end

        `ifdef FLEN_64
            3'b101: // c.fsdsp
                instr_o = {sdsp_imm[11:5], rs2, 5'd2, 3'b011, sdsp_imm[4:0], `INST_FS};
        `endif

            3'b110:   // c.swsp
                instr_o = {swsp_imm[11:5], rs2, 5'd2, 3'b010, swsp_imm[4:0], `INST_S};

        `ifdef XLEN_64
            3'b111:   // c.sdsp
                instr_o = {sdsp_imm[11:5], rs2, 5'd2, 3'b011, sdsp_imm[4:0], `INST_S};
        `else
            3'b111: // c.fswsp
                instr_o = {swsp_imm[11:5], rs2, 5'd2, 3'b010, swsp_imm[4:0], `INST_FS};
        `endif

            default:
                instr_o = 32'h00000013;
            endcase
        end

        default:
            instr_o = 32'h00000013;
        endcase

        return instr_o;
    endfunction

    function automatic [31:0] decompress32 (input [31:0] insn);
        return insn;
    endfunction

    //-------------------------------------------------------------------------
    // Input ready: we can accept a new word when we don't have a word
    // already buffered that still needs to emit instructions.
    //
    // This is a conservative template (simple to understand) and can be
    // later optimized to overlap more.
    //-------------------------------------------------------------------------

    assign ready_in = ~have_word && (~valid_out || (valid_out && ready_out));

    //-------------------------------------------------------------------------
    // Main state machine
    //-------------------------------------------------------------------------

    always @(posedge clk) begin
        if (reset) begin
            have_word       <= 0;

            valid_out       <= 0;
            instr_out       <= '0;
            PC_out          <= '0;
            tmask_out       <= '0;
            wid_out         <= '0;
            uuid_out        <= '0;
            state_out       <= '0;

            rvc_out         <= 0;
            incomplete_out  <= 0;

        end else begin

            rvc_out        <= 1'b0; 
            incomplete_out <= 1'b0;

            // PC_out        <= PC_in;
            // tmask_out     <= tmask_in;
            // wid_out       <= wid_in;
            // uuid_out      <= uuid_in;
            // state_out     <= state_in;
            //state_out      <= state_reg; 

            // Default: if downstream consumed the current instruction, drop valid_out
            if (valid_out && ready_out)
                valid_out <= 0;

            // Accept a new icache word when allowed
            if (valid_in && ready_in) begin
                word_reg  <= word_in;
                pc_reg    <= PC_in;
                tmask_reg <= tmask_in;
                wid_reg   <= wid_in;
                uuid_reg  <= uuid_in;
                state_reg <= state_in;
                have_word <= 1;
            end
            
            if(~valid_out) begin
                // Priority 1: if we have a leftover half and no output in flight, emit that as the next instruction.
                if (state_reg == 2'b01) begin
                    instr_out     <= decompress16(half_reg[wid_reg]);
                    PC_out        <= pc_reg;
                    tmask_out     <= tmask_reg;
                    wid_out       <= wid_reg;
                    uuid_out      <= uuid_reg;
                    state_out     <= 2'b00;
                    valid_out     <= 1;

                    have_word     <= 0;
                    rvc_out       <= 1'b1;
                    incomplete_out<= 1'b0;

                    state_reg     <= 2'b00;
                end

                // --- Priority 2: finishing a 32-bit instr spanning words ---
                else if (have_word && state_reg == 2'b10) begin
                    logic [31:0] new_word;
                    logic [15:0] low_half_new;
                    logic [15:0] high_half_new;
                    logic        new_high_is_c;

                    new_word      = word_reg[31:0];
                    low_half_new  = new_word[15:0];
                    high_half_new  = new_word[31:16];
                    new_high_is_c = is_rvc(high_half_new[1:0]);

                    instr_out     <= decompress32({low_half_new, half_reg[wid_reg]});
                    PC_out        <= pc_reg;
                    tmask_out     <= tmask_reg;
                    wid_out       <= wid_reg;
                    uuid_out      <= uuid_reg;
                    state_out     <= new_high_is_c ? 2'b01 : 2'b10;
                    valid_out     <= 1;

                    half_reg[wid_reg] <= high_half_new;

                    have_word     <= 0;
                    rvc_out       <= 1'b0;
                    incomplete_out<= 1'b0;

                    state_reg     <= state_out;
                end

                // --- Priority 3: decode from current word_reg ---
                else if (have_word && state_reg == 2'b00) begin
                    logic [31:0] full_word;
                    logic [15:0] low_half, high_half;
                    logic        low_is_c, high_is_c;
                    logic [1:0]  pc_lsb;

                    full_word   = word_reg[31:0];
                    low_half    = full_word[15:0];
                    high_half   = full_word[31:16];
                    low_is_c    = is_rvc(low_half[1:0]);
                    high_is_c   = is_rvc(high_half[1:0]);
                    pc_lsb      = pc_reg[1:0];

                    // Decide where the instruction starts based on PC[1:0]
                    unique case (pc_lsb)
                        // ---------------- PC[1:0] == 2'b00 ----------------
                        2'b00: begin
                            if (low_is_c && high_is_c) begin
                                // C + C: emit low now, save high for next cycle
                                instr_out        <= decompress16(low_half);
                                PC_out           <= pc_reg;
                                tmask_out        <= tmask_reg;
                                wid_out          <= wid_reg;
                                uuid_out         <= uuid_reg;
                                state_out        <= 2'b01;
                                valid_out        <= 1;

                                half_reg[wid_reg] <= high_half;

                                have_word        <= 0;
                                rvc_out          <= 1'b1;
                                incomplete_out   <= 1'b0;

                            end else if (low_is_c) begin
                                // C + (something else) - we just emit the first C
                                instr_out        <= decompress16(low_half);
                                PC_out           <= pc_reg;
                                tmask_out        <= tmask_reg;
                                wid_out          <= wid_reg;
                                uuid_out         <= uuid_reg;
                                state_out        <= 2'b10;
                                valid_out        <= 1;

                                half_reg[wid_reg] <= high_half;

                                have_word        <= 0;
                                rvc_out          <= 1'b1;
                                incomplete_out   <= 1'b0;
                            end else begin
                                // Non-compressed 32-bit at low_half
                                instr_out        <= decompress32(full_word);
                                PC_out           <= pc_reg;
                                tmask_out        <= tmask_reg;
                                wid_out          <= wid_reg;
                                uuid_out         <= uuid_reg;
                                state_out        <= 2'b00;
                                valid_out        <= 1;

                                have_word        <= 0;
                                rvc_out          <= 1'b0;
                                incomplete_out   <= 1'b0;
                            end
                        end

                        // ---------------- PC[1:0] == 2'b10 ----------------
                        2'b10: begin
                            // Start at high_half
                            if (high_is_c) begin
                                // Compressed at high_half: straightforward
                                instr_out      <= decompress16(high_half);
                                PC_out         <= pc_reg;
                                tmask_out      <= tmask_reg;
                                wid_out        <= wid_reg;
                                uuid_out       <= uuid_reg;
                                state_out      <= 2'b00;
                                valid_out      <= 1;

                                have_word      <= 0;
                                rvc_out        <= 1'b1;
                                incomplete_out <= 1'b0;

                            end else begin
                                // Save upper16 now, wait for the next word to supply
                                // the low16 in the Priority 2 block above.
                                instr_out      <= '0; //This should not matter
                                PC_out         <= pc_reg;
                                tmask_out      <= tmask_reg;
                                wid_out        <= wid_reg;
                                uuid_out       <= uuid_reg;
                                state_out      <= 2'b10;
                                
                                half_reg[wid_reg] <= high_half;
                                // Don't emit anything yet (valid_out remains 0)
                                // Mark this word as used.
                                have_word      <= 0;
                                rvc_out        <= 1'b0;
                                incomplete_out <= 1'b1;
                            end
                        end

                        // ---------------- Other alignments ----------------
                        default: begin
                            instr_out        <= decompress32(full_word);
                            PC_out           <= pc_reg;
                            tmask_out        <= tmask_reg;
                            wid_out          <= wid_reg;
                            uuid_out         <= uuid_reg;
                            state_out        <= 2'b00;
                            valid_out        <= 1;

                            have_word        <= 0;
                            rvc_out          <= 1'b0;
                            incomplete_out   <= 1'b0;
                        end
                    endcase
                end
            end
        end
    end
    //`RUNTIME_ASSERT(~PC_misaligned, ("PC misaligned!!"))

`ifdef DBG_TRACE_PIPELINE

    integer decomp_cnt;

    always_ff @(posedge clk) begin
        if (reset) begin
            decomp_cnt <= 0;
        end else if (valid_out && ready_out) begin
            decomp_cnt <= decomp_cnt + 1;
            $display("[RTL  C%3d] PC=0x%08h in32=0x%08h out32=0x%08h state=%2b wid=%2d",
                    decomp_cnt,
                    PC_out,
                    word_reg,
                    instr_out,
                    state_out,
                    wid_out);
        end
    end
`endif
endmodule
