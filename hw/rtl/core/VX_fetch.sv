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

`include "VX_define.vh"

module VX_fetch import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input  wire             clk,
    input  wire             reset,

    // Icache interface
    VX_mem_bus_if.master    icache_bus_if,

    // inputs
    VX_schedule_if.slave    schedule_if,

    // outputs
    VX_fetch_if.master      fetch_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    //`UNUSED_VAR (reset)

    VX_decompressor #(
        .INSTANCE_ID (INSTANCE_ID)
    ) decompressor_u (
        .clk          (clk),
        .reset        (reset),
        .icache_bus_if(icache_bus_if),
        .schedule_if  (schedule_if),
        .fetch_out_if (fetch_if)
    );


`ifdef SCOPE
`ifdef DBG_SCOPE_FETCH
    `SCOPE_IO_SWITCH (1);
    wire schedule_fire = schedule_if.valid && schedule_if.ready;
    wire icache_bus_req_fire = icache_bus_if.req_valid && icache_bus_if.req_ready;
    wire icache_bus_rsp_fire = icache_bus_if.rsp_valid && icache_bus_if.rsp_ready;
    wire reset_negedge;
    `NEG_EDGE (reset_negedge, reset);
    `SCOPE_TAP_EX (0, 1, 6, 3, (
            UUID_WIDTH + NW_WIDTH + `NUM_THREADS + PC_BITS +
            UUID_WIDTH + ICACHE_WORD_SIZE + ICACHE_ADDR_WIDTH +
            UUID_WIDTH + (ICACHE_WORD_SIZE * 8)
        ), {
            schedule_if.valid,
            schedule_if.ready,
            icache_bus_if.req_valid,
            icache_bus_if.req_ready,
            icache_bus_if.rsp_valid,
            icache_bus_if.rsp_ready
        }, {
            schedule_fire,
            icache_bus_req_fire,
            icache_bus_rsp_fire
        },{
            schedule_if.data.uuid, schedule_if.data.wid, schedule_if.data.tmask, schedule_if.data.PC,
            icache_bus_if.req_data.tag.uuid, icache_bus_if.req_data.byteen, icache_bus_if.req_data.addr,
            icache_bus_if.rsp_data.tag.uuid, icache_bus_if.rsp_data.data
        },
        reset_negedge, 1'b0, 4096
    );
`else
    `SCOPE_IO_UNUSED(0)
`endif
`endif

`ifdef CHIPSCOPE
`ifdef DBG_SCOPE_FETCH
    ila_fetch ila_fetch_inst (
        .clk    (clk),
        .probe0 ({schedule_if.valid, schedule_if.data, schedule_if.ready}),
        .probe1 ({icache_bus_if.req_valid, icache_bus_if.req_data, icache_bus_if.req_ready}),
        .probe2 ({icache_bus_if.rsp_valid, icache_bus_if.rsp_data, icache_bus_if.rsp_ready})
    );
`endif
`endif

`ifdef DBG_TRACE_MEM
    always @(posedge clk) begin
        if (schedule_if.valid && schedule_if.ready) begin
            `TRACE(1, ("%t: %s req: wid=%0d, PC=0x%0h, tmask=%b (#%0d)\n", $time, INSTANCE_ID, schedule_if.data.wid, to_fullPC(schedule_if.data.PC), schedule_if.data.tmask, schedule_if.data.uuid))
        end
        if (fetch_if.valid && fetch_if.ready) begin
            `TRACE(1, ("%t: %s rsp: wid=%0d, PC=0x%0h, tmask=%b, instr=0x%0h (#%0d)\n", $time, INSTANCE_ID, fetch_if.data.wid, to_fullPC(fetch_if.data.PC), fetch_if.data.tmask, fetch_if.data.word, fetch_if.data.uuid))
        end
    end
`endif

endmodule
