`include "VX_define.vh"

module VX_decompressor import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input  wire             clk,
    input  wire             reset,

    // Icache interface (owned by decompressor)
    VX_mem_bus_if.master    icache_bus_if,

    // input from scheduler (real byte PC like emulator warp.PC)
    VX_schedule_if.slave    schedule_if,

    // output to decode
    VX_fetch_if.master      fetch_out_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // ------------------------------------------------------------------------
    // I$ request / response meta
    // ------------------------------------------------------------------------

    wire icache_req_valid;
    wire icache_req_ready;

    wire [UUID_WIDTH-1:0] rsp_uuid;
    wire [NW_WIDTH-1:0]   rsp_tag;

    assign {rsp_uuid, rsp_tag} = icache_bus_if.rsp_data.tag;

    // Store (PC, tmask) per warp for each I$ request so we can reconstruct on rsp
    wire [PC_BITS-1:0]       rsp_PC;
    wire [`NUM_THREADS-1:0]  rsp_tmask;

    VX_dp_ram #(
        .DATAW   (PC_BITS + `NUM_THREADS),
        .SIZE    (`NUM_WARPS),
        .RDW_MODE("R"),
        .LUTRAM  (1)
    ) tag_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (icache_req_fire),
        .wren  (1'b1),
        .waddr (ic_req_wid),
        .wdata ({ic_req_PC, ic_req_tmask}),
        .raddr (rsp_tag),
        .rdata ({rsp_PC, rsp_tmask})
    );

    // Scheduler-side / follow-up side request selection
    logic                        sched_req_valid;
    logic [ICACHE_ADDR_WIDTH-1:0] sched_req_addr;
    logic [ICACHE_TAG_WIDTH-1:0]  sched_req_tag;
    logic [PC_BITS-1:0]           sched_req_PC;
    logic [`NUM_THREADS-1:0]      sched_req_tmask;
    logic [NW_WIDTH-1:0]          sched_req_wid;
    logic [UUID_WIDTH-1:0]        sched_req_uuid;

    // follow-up request (for BUF_32HI second word)
    logic                        follow_req_valid;
    logic [ICACHE_ADDR_WIDTH-1:0] follow_req_addr;
    logic [ICACHE_TAG_WIDTH-1:0]  follow_req_tag;
    logic [PC_BITS-1:0]           follow_req_PC;
    logic [`NUM_THREADS-1:0]      follow_req_tmask;
    logic [NW_WIDTH-1:0]          follow_req_wid;
    logic [UUID_WIDTH-1:0]        follow_req_uuid;

    // selected request (follow-up has priority)
    logic                        ic_req_valid;
    logic [ICACHE_ADDR_WIDTH-1:0] ic_req_addr;
    logic [ICACHE_TAG_WIDTH-1:0]  ic_req_tag;
    logic [PC_BITS-1:0]           ic_req_PC;
    logic [`NUM_THREADS-1:0]      ic_req_tmask;
    logic [NW_WIDTH-1:0]          ic_req_wid;

    // fire = selected request accepted by req_buf
    wire icache_req_fire = ic_req_valid && icache_req_ready;

`ifndef L1_ENABLE
    // ibuffer fullness protection (same as Vortex)
    wire [`NUM_WARPS-1:0] pending_ibuf_full;
    for (genvar i = 0; i < `NUM_WARPS; ++i) begin : g_pending_reads
        VX_pending_size #(
            .SIZE (`IBUF_SIZE)
        ) pending_reads (
            .clk   (clk),
            .reset (reset),
            .incr  (icache_req_fire && (ic_req_wid == i)),
            .decr  (fetch_out_if.ibuf_pop[i]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (alm_empty),
            .full  (pending_ibuf_full[i]),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );
    end

    wire ibuf_ready = ~pending_ibuf_full[schedule_if.data.wid];
`else
    wire ibuf_ready = 1'b1;
`endif

    `RUNTIME_ASSERT((!schedule_if.valid || schedule_if.data.PC != 0),
        ("%t: *** %s invalid PC=0x%0h, wid=%0d, tmask=%b (#%0d)",
            $time, INSTANCE_ID,
            to_fullPC(schedule_if.data.PC),
            schedule_if.data.wid,
            schedule_if.data.tmask,
            schedule_if.data.uuid))

    // Scheduler-side meta (real byte PC)
    assign sched_req_uuid  = schedule_if.data.uuid;
    assign sched_req_wid   = schedule_if.data.wid;
    assign sched_req_PC    = schedule_if.data.PC;
    assign sched_req_tmask = schedule_if.data.tmask;

    // First-word request from scheduler
    assign sched_req_valid = schedule_if.valid && ibuf_ready;

    // word-aligned I$ addr = (PC & ~3) >> 2
    assign sched_req_addr  = to_fullPC(sched_req_PC)[ICACHE_ADDR_WIDTH+1 : 2];
    assign sched_req_tag   = {sched_req_uuid, sched_req_wid};

    // Select between scheduler request and follow-up request.
    // Follow-up has priority if asserted.
    assign ic_req_valid  = follow_req_valid ? follow_req_valid : sched_req_valid;
    assign ic_req_addr   = follow_req_valid ? follow_req_addr  : sched_req_addr;
    assign ic_req_tag    = follow_req_valid ? follow_req_tag   : sched_req_tag;
    assign ic_req_PC     = follow_req_valid ? follow_req_PC    : sched_req_PC;
    assign ic_req_tmask  = follow_req_valid ? follow_req_tmask : sched_req_tmask;
    assign ic_req_wid    = follow_req_valid ? follow_req_wid   : sched_req_wid;

    // I$ sees this from elastic buffer
    assign icache_req_valid = ic_req_valid;

    // Scheduler is only "ready" when its request is the one being driven
    assign schedule_if.ready = icache_req_ready && ibuf_ready && ~follow_req_valid;

    // Elastic buffer in front of external I$ bus
    VX_elastic_buffer #(
        .DATAW   (ICACHE_ADDR_WIDTH + ICACHE_TAG_WIDTH),
        .SIZE    (2),
        .OUT_REG (1) // external bus registered
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (icache_req_valid),
        .ready_in  (icache_req_ready),
        .data_in   ({ic_req_addr, ic_req_tag}),
        .data_out  ({icache_bus_if.req_data.addr, icache_bus_if.req_data.tag}),
        .valid_out (icache_bus_if.req_valid),
        .ready_out (icache_bus_if.req_ready)
    );

    assign icache_bus_if.req_data.flags  = '0;
    assign icache_bus_if.req_data.rw     = 1'b0;
    assign icache_bus_if.req_data.byteen = '1;
    assign icache_bus_if.req_data.data   = '0;

    // I$ response → internal fetch interface
    VX_fetch_if fetch_in_if();

    assign fetch_in_if.valid       = icache_bus_if.rsp_valid;
    assign fetch_in_if.data.tmask  = rsp_tmask;
    assign fetch_in_if.data.wid    = rsp_tag;
    assign fetch_in_if.data.PC     = rsp_PC;                  // real byte PC for this word
    assign fetch_in_if.data.word   = icache_bus_if.rsp_data.data; // raw 32b word
    assign fetch_in_if.data.uuid   = rsp_uuid;
    assign fetch_in_if.data.last_in_word = 1'b1;  // default, refined in FSM

    // I$ ready when decompressor is ready
    assign icache_bus_if.rsp_ready = fetch_in_if.ready;

`ifndef L1_ENABLE
    // pass ibuf_pop through
    assign fetch_in_if.ibuf_pop = fetch_out_if.ibuf_pop;
`endif

    // ------------------------------------------------------------------------
    // 16-bit -> 32-bit decompressor (pure combinational, same as your code)
    // ------------------------------------------------------------------------

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

    function automatic logic [31:0] decompress16 (
        input logic [15:0] instr_i
    );
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

    // compressed if [1:0] != 2'b11
    function automatic logic is_rvc16 (input logic [1:0] op);
        return (op != 2'b11);
    endfunction

    // ------------------------------------------------------------------------
    // Halfword buffer per warp (RVC / cross-word 32b), emulator-style
    // ------------------------------------------------------------------------

    RVC_data_t buffer   [`NUM_WARPS];
    RVC_data_t buffer_n [`NUM_WARPS];

    // ------------------------------------------------------------------------
    // Main combinational control (state machine over halfwords)
    // ------------------------------------------------------------------------

    always_comb begin : decomp_fsm
        // Local temps
        logic        in_valid;
        fetch_t      in_data;
        logic        out_ready;
        logic [31:0] word, word2;
        logic [15:0] low16, high16, low2, high2;
        logic        low_is_c, high_is_c;
        logic        pc_low;

        logic [NW_WIDTH-1:0] cur_wid;
        buf_state_e          cur_state;

        // Look for any warp that has a buffered RVC
        logic                have_rvc;
        logic [NW_WIDTH-1:0] rvc_wid;

        // PC helpers
        logic [`XLEN-1:0] inst_full_pc;
        logic [`XLEN-1:0] base_full_pc;
        logic [`XLEN-1:0] next_full_pc;

        logic [`XLEN-1:0] inst_full_pc2;
        logic [`XLEN-1:0] base_full_pc2;
        logic [`XLEN-1:0] next_full_pc2;

        // ----------------------------------------
        // Basic input / output defaults
        // ----------------------------------------
        in_valid  = fetch_in_if.valid;
        in_data   = fetch_in_if.data;
        out_ready = fetch_out_if.ready;

        pc_low  = (to_fullPC(in_data.PC)[1] == 1'b0);

        word      = '0;
        word2     = '0;
        low16     = '0;
        high16    = '0;
        low2      = '0;
        high2     = '0;
        low_is_c  = 1'b0;
        high_is_c = 1'b0;

        have_rvc  = 1'b0;
        rvc_wid   = '0;

        cur_wid   = '0;
        cur_state = BUF_EMPTY;

        // Default outputs
        fetch_out_if.valid           = 1'b0;
        fetch_out_if.data            = '0;
        fetch_out_if.data.last_in_word = in_data.last_in_word;  // default pass-through
        fetch_in_if.ready            = 1'b0;

        // Default: no scheduler feedback
        schedule_if.decompress_finished = 1'b0;
        schedule_if.pc_incr_by_2        = 1'b0;
        schedule_if.decompress_wid      = '0;

        // Default: no follow-up request this cycle
        follow_req_valid  = 1'b0;
        follow_req_addr   = '0;
        follow_req_tag    = '0;
        follow_req_PC     = '0;
        follow_req_tmask  = '0;
        follow_req_wid    = '0;
        follow_req_uuid   = '0;

        // default PC helper temporaries
        inst_full_pc  = '0;
        base_full_pc  = '0;
        next_full_pc  = '0;
        inst_full_pc2 = '0;
        base_full_pc2 = '0;
        next_full_pc2 = '0;

        // default: keep all warp-local buffers as-is
        for (int w = 0; w < `NUM_WARPS; ++w) begin
            buffer_n[w] = buffer[w];
        end

        // ----------------------------------------
        // Flush stale buffers when scheduler PC changes for a warp
        // ----------------------------------------
        if (sched_req_valid) begin
            for (int w = 0; w < `NUM_WARPS; ++w) begin
                if ((sched_req_wid == w[NW_WIDTH-1:0])
                 && (buffer[w].state != BUF_EMPTY)
                 && (buffer[w].pc    != sched_req_PC)) begin
                    buffer_n[w].state = BUF_EMPTY;
                end
            end
        end

        // ----------------------------------------
        // Find any warp with a buffered RVC (post-flush)
        // ----------------------------------------
        for (int w = 0; w < `NUM_WARPS; ++w) begin
            if (!have_rvc && (buffer_n[w].state == BUF_RVC)) begin
                have_rvc = 1'b1;
                rvc_wid  = w[NW_WIDTH-1:0];
            end
        end

        if (have_rvc) begin
            cur_wid   = rvc_wid;
            cur_state = buffer_n[cur_wid].state;
        end else begin
            cur_wid   = in_data.wid;
            cur_state = buffer_n[cur_wid].state;
        end

        // ========================================
        // State machine
        // ========================================

        // =========================
        // Case 1: buffered RVC inst
        // =========================
        if (cur_state == BUF_RVC) begin
            fetch_out_if.valid         = 1'b1;
            fetch_out_if.data.uuid     = buffer[cur_wid].uuid;
            fetch_out_if.data.wid      = cur_wid;
            fetch_out_if.data.tmask    = buffer[cur_wid].tmask;
            fetch_out_if.data.PC       = buffer[cur_wid].pc;
            fetch_out_if.data.word     = decompress16(buffer[cur_wid].hw);
            fetch_out_if.data.last_in_word = 1'b1; // finishes that word

            // do not consume a new I$ word in this cycle
            fetch_in_if.ready = 1'b0;

            if (out_ready) begin
                buffer_n[cur_wid].state = BUF_EMPTY;

                // ---- feedback to scheduler ----
                schedule_if.decompress_finished = 1'b1;
                schedule_if.decompress_wid      = cur_wid;
                schedule_if.pc_incr_by_2        = 1'b1; // RVC → PC += 2
            end

        // ==============================
        // Case 2: buffered high half of 32b
        // ==============================
        end else if (cur_state == BUF_32HI) begin
            if (in_valid) begin
                word2 = in_data.word;
                low2  = word2[15:0];
                high2 = word2[31:16];

                // Must not be compressed: low2 is the low half of the 32b inst
                //`RUNTIME_ASSERT(!is_rvc16(low2[1:0]), "illegal pattern: BUF_32HI but low2 looks compressed!")
                //`RUNTIME_ASSERT(!is_rvc16(low2[1:0]),
                //    ("%t: *** %s illegal pattern: BUF_32HI but low2 looks compressed! "
                //     "wid=%0d PC_prev=0x%0h word2=0x%08h",
                //     $time, INSTANCE_ID, cur_wid, to_fullPC(buf_pc[cur_wid]), word2))

                fetch_out_if.valid      = 1'b1;
                fetch_out_if.data.uuid  = buffer[cur_wid].uuid;
                fetch_out_if.data.wid   = cur_wid;
                fetch_out_if.data.tmask = buffer[cur_wid].tmask;
                fetch_out_if.data.PC    = buffer[cur_wid].pc; // PC of 32b instr
                fetch_out_if.data.word  = {low2, buffer[cur_wid].hw};
                // last_in_word decided below

                if (out_ready) begin
                    fetch_in_if.ready = 1'b1;

                    // After finishing this 32b, keep high2 as buffered halfword.
                    if (is_rvc16(high2[1:0])) begin
                        // high2 is compressed → BUF_RVC at PC + 4
                        buffer_n[cur_wid].state  = BUF_RVC;
                        buffer_n[cur_wid].hw     = high2;
                        buffer_n[cur_wid].pc     = from_fullPC(
                            to_fullPC(buffer[cur_wid].pc) + `XLEN'(4)
                        );
                        buffer_n[cur_wid].uuid   = in_data.uuid;
                        buffer_n[cur_wid].tmask  = in_data.tmask;
                        fetch_out_if.data.last_in_word = 1'b0;
                    end else begin
                        // new cross-boundary 32b starting at PC+4
                        buffer_n[cur_wid].state  = BUF_32HI;
                        buffer_n[cur_wid].hw     = high2;
                        buffer_n[cur_wid].pc     = from_fullPC(
                            to_fullPC(buffer[cur_wid].pc) + `XLEN'(4)
                        );
                        buffer_n[cur_wid].uuid   = in_data.uuid;
                        buffer_n[cur_wid].tmask  = in_data.tmask;
                        fetch_out_if.data.last_in_word = 1'b0;

                        // Need second word for this 32b
                        inst_full_pc2 = to_fullPC(buffer[cur_wid].pc) + `XLEN'(4);
                        base_full_pc2 = inst_full_pc2 & ~`XLEN'(3);
                        next_full_pc2 = base_full_pc2 + `XLEN'(4);

                        follow_req_valid  = 1'b1;
                        follow_req_PC     = from_fullPC(next_full_pc2);
                        follow_req_tmask  = in_data.tmask;
                        follow_req_wid    = cur_wid;
                        follow_req_uuid   = in_data.uuid;
                        follow_req_addr   = next_full_pc2[ICACHE_ADDR_WIDTH+1 : 2];
                        follow_req_tag    = {follow_req_uuid, follow_req_wid};
                    end

                    // ---- feedback to scheduler ----
                    schedule_if.decompress_finished = 1'b1;
                    schedule_if.decompress_wid      = cur_wid;
                    schedule_if.pc_incr_by_2        = 1'b0; // full 32b → PC += 4
                end
            end

        // ======================
        // Case 3: no buffered hw
        // ======================
        end else begin : buf_empty_block // BUF_EMPTY
            if (in_valid) begin
                word   = in_data.word;
                low16  = word[15:0];
                high16 = word[31:16];

                fetch_out_if.data.uuid  = in_data.uuid;
                fetch_out_if.data.wid   = in_data.wid;
                fetch_out_if.data.tmask = in_data.tmask;
                fetch_out_if.data.PC    = in_data.PC;

                // we will set valid when we emit an instruction
                fetch_out_if.valid = 1'b1;

                if (out_ready) begin
                    fetch_in_if.ready = 1'b1;

                    if (pc_low) begin
                        // ===========================
                        // Case 3a: PC points to low16
                        // ===========================
                        low_is_c = is_rvc16(low16[1:0]);

                        if (low_is_c) begin
                            // low16 is RVC
                            fetch_out_if.data.word = decompress16(low16);

                            if (is_rvc16(high16[1:0])) begin
                                // next RVC in same word
                                buffer_n[in_data.wid].state  = BUF_RVC;
                                buffer_n[in_data.wid].hw     = high16;
                                buffer_n[in_data.wid].pc     = from_fullPC(
                                    to_fullPC(in_data.PC) + `XLEN'(2)
                                );
                                buffer_n[in_data.wid].uuid   = in_data.uuid;
                                buffer_n[in_data.wid].tmask  = in_data.tmask;
                                fetch_out_if.data.last_in_word = 1'b0;
                            end else begin
                                // cross-boundary 32b starting at high16
                                buffer_n[in_data.wid].state  = BUF_32HI;
                                buffer_n[in_data.wid].hw     = high16;
                                buffer_n[in_data.wid].pc     = from_fullPC(
                                    to_fullPC(in_data.PC) + `XLEN'(2)
                                );
                                buffer_n[in_data.wid].uuid   = in_data.uuid;
                                buffer_n[in_data.wid].tmask  = in_data.tmask;

                                fetch_out_if.data.last_in_word = 1'b0;

                                // Need second word for this 32b:
                                inst_full_pc = to_fullPC(in_data.PC) + `XLEN'(2);
                                base_full_pc = inst_full_pc & ~`XLEN'(3);
                                next_full_pc = base_full_pc + `XLEN'(4);

                                follow_req_valid  = 1'b1;
                                follow_req_PC     = from_fullPC(next_full_pc);
                                follow_req_tmask  = in_data.tmask;
                                follow_req_wid    = in_data.wid;
                                follow_req_uuid   = in_data.uuid;
                                follow_req_addr   = next_full_pc[ICACHE_ADDR_WIDTH+1 : 2];
                                follow_req_tag    = {follow_req_uuid, follow_req_wid};
                            end

                            // ---- feedback to scheduler ----
                            schedule_if.decompress_finished = 1'b1;
                            schedule_if.decompress_wid      = in_data.wid;
                            schedule_if.pc_incr_by_2        = 1'b1; // RVC → +2

                        end else begin
                            // 32b fully inside this word at low16
                            fetch_out_if.data.word        = word;
                            buffer_n[in_data.wid].state      = BUF_EMPTY;
                            fetch_out_if.data.last_in_word = 1'b1;

                            // ---- feedback to scheduler ----
                            schedule_if.decompress_finished = 1'b1;
                            schedule_if.decompress_wid      = in_data.wid;
                            schedule_if.pc_incr_by_2        = 1'b0; // full 32b → +4
                        end

                    end else begin
                        // ================================
                        // Case 3b: PC points to high16
                        // ================================
                        high_is_c = is_rvc16(high16[1:0]);

                        if (high_is_c) begin
                            // high16 is RVC
                            fetch_out_if.data.word        = decompress16(high16);
                            buffer_n[in_data.wid].state      = BUF_EMPTY;
                            fetch_out_if.data.last_in_word = 1'b1;

                            // ---- feedback to scheduler ----
                            schedule_if.decompress_finished = 1'b1;
                            schedule_if.decompress_wid      = in_data.wid;
                            schedule_if.pc_incr_by_2        = 1'b1; // RVC → +2

                        end else begin
                            // high16 is upper half of 32b crossing a boundary
                            buffer_n[in_data.wid].state  = BUF_32HI;
                            buffer_n[in_data.wid].hw     = high16;
                            buffer_n[in_data.wid].pc     = in_data.PC;   // PC at high16
                            buffer_n[in_data.wid].uuid   = in_data.uuid;
                            buffer_n[in_data.wid].tmask  = in_data.tmask;
                            // We DO NOT emit an instruction this cycle
                            fetch_out_if.valid = 1'b0;
                            fetch_out_if.data  = '0;

                            inst_full_pc = to_fullPC(in_data.PC); // PC at high16
                            base_full_pc = inst_full_pc & ~`XLEN'(3);
                            next_full_pc = base_full_pc + `XLEN'(4);

                            follow_req_valid  = 1'b1;
                            follow_req_PC     = from_fullPC(next_full_pc);
                            follow_req_tmask  = in_data.tmask;
                            follow_req_wid    = in_data.wid;
                            follow_req_uuid   = in_data.uuid;
                            follow_req_addr   = next_full_pc[ICACHE_ADDR_WIDTH+1 : 2];
                            follow_req_tag    = {follow_req_uuid, follow_req_wid};
                        end
                    end
                end else begin
                    // out_ready == 0 : hold valid, don't consume input
                end
            end else begin
                // no in_valid and no BUF_RVC/BUF_32HI: idle
                fetch_out_if.valid = 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------------
    // Sequential update of buffer regs
    // ------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            for (int w = 0; w < `NUM_WARPS; ++w) begin
                buffer[w].state  <= BUF_EMPTY;
                buffer[w].hw     <= '0;
                buffer[w].pc     <= '0;
                buffer[w].uuid   <= '0;
                buffer[w].tmask  <= '0;
            end
        end else begin
            for (int w = 0; w < `NUM_WARPS; ++w) begin
                buffer[w].state  <= buffer_n[w].state;
                buffer[w].hw     <= buffer_n[w].hw;
                buffer[w].pc     <= buffer_n[w].pc;
                buffer[w].uuid   <= buffer_n[w].uuid;
                buffer[w].tmask  <= buffer_n[w].tmask;
            end
        end
    end

`ifdef DBG_TRACE_PIPELINE
    integer decomp_cnt;

    always_ff @(posedge clk) begin
        if (reset) begin
            decomp_cnt <= 0;
        end else if (fetch_out_if.valid && fetch_out_if.ready) begin
            if (fetch_out_if.data.wid == 2'b10) begin
                decomp_cnt <= decomp_cnt + 1;
                $display("[RTL  C%3d] PC=%08h out32=%08h lo16=%04h hi16=%04h wid=%0d",
                        decomp_cnt,
                        fetch_out_if.data.PC,
                        fetch_out_if.data.word,
                        fetch_out_if.data.word[15:0],
                        fetch_out_if.data.word[31:16],
                        fetch_out_if.data.wid);
            end
        end
    end
`endif

endmodule
