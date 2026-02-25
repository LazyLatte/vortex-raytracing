#pragma once
#include <simobject.h>
#include "arch.h"
#include "dcrs.h"
#include "rt_trace.h"
#include <vector>

namespace vortex {

class Core;
class RTSim;

class RTUnit : public SimObject<RTUnit> {
public:
	struct PerfStats {
        unsigned long long rt_active_cycles;
        unsigned long long total_elapsed_cycles;
        // uint64_t rt_mem_store_q_cycles;
        unsigned long long rt_total_warp_latency;
        unsigned long long rt_total_thread_latency;
        double rt_total_simt_efficiency;
        // double rt_total_warp_occupancy;
        unsigned rt_total_warps;
        unsigned rt_latency_counter;
        unsigned long long rt_latency_dist[warp_statuses][ray_statuses] = {};

        PerfStats()
            : rt_active_cycles(0)
            , total_elapsed_cycles(0)
            , rt_total_warp_latency(0)
            , rt_total_thread_latency(0)
            , rt_total_simt_efficiency(0.0)
            , rt_total_warps(0)
            , rt_latency_counter(0)
        {}

        void add_rt_latency_dist(unsigned *update) {
            for (unsigned i=0; i<warp_statuses; i++) {
                for (unsigned j=0; j<ray_statuses; j++) {
                    rt_latency_dist[i][j] += update[i*ray_statuses + j];
                }
            }
            rt_latency_counter++;
        }
	};
  
    std::vector<SimChannel<instr_trace_t*>> Inputs; 
    std::vector<SimChannel<instr_trace_t*>> Outputs; 

    std::vector<std::vector<SimChannel<MemReq>>> rtu_dcache_req_out;
    std::vector<std::vector<SimChannel<MemRsp>>> rtu_dcache_rsp_in;
    RTUnit(const SimContext &ctx, const char* name, const Arch &arch, const DCRS &dcrs, Core* core);
    ~RTUnit();
    void reset();
    void tick();

    void dcache_read(void* data, uint64_t addr, uint32_t size);
    void dcache_write(const void* data, uint64_t addr, uint32_t size);

    void init_ray(std::vector<reg_data_t>& rd_data);
    void set_ray_properties(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data, uint32_t axis);
    void set_payload_addr(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data);

    void traverse(const std::vector<reg_data_t>& rs1_data, RtuTraceData* trace_data);
    void get_work(std::vector<reg_data_t>& rd_data);
    void get_attr(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, std::vector<reg_data_t>& rd_data);
    void commit(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, RtuTraceData* trace_data);

    void print_stats() const;
    const PerfStats& perf_stats() const;
private:
	class Impl;
	Impl* impl_;
};

}