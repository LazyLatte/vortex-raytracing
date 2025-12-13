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

interface VX_schedule_if import VX_gpu_pkg::*; ();

    logic  valid;
    schedule_t data;
    logic  ready;
    logic decompress_finished;
    logic pc_incr_by_2;
    logic [NW_WIDTH-1:0] decompress_wid;

    modport master (
        output valid,
        output data,
        input  ready,
        input decompress_finished,
        input pc_incr_by_2,
        input decompress_wid
    );

    modport slave (
        input  valid,
        input  data,
        output ready,
        output decompress_finished,
        output pc_incr_by_2,
        output decompress_wid
    );

endinterface
