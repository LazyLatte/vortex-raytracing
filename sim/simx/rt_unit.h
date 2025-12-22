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
        // uint64_t rt_total_cycles;
        // uint64_t rt_mem_store_q_cycles;
        unsigned long long rt_total_warp_latency;
        unsigned long long rt_total_thread_latency;
        double rt_total_simt_efficiency;
        // double rt_total_warp_occupancy;
        unsigned rt_total_warps;
        unsigned rt_latency_counter;
        unsigned long long rt_latency_dist[warp_statuses][ray_statuses] = {};

        PerfStats()
            : rt_total_warp_latency(0)
            , rt_total_thread_latency(0)
            , rt_total_simt_efficiency(0.0)
            , rt_total_warps(0)
            , rt_latency_counter(0)
        {}

        // PerfStats& operator+=(const PerfStats& rhs) {
        //     this->reads   += rhs.reads;
        //     this->latency += rhs.latency;
        //     this->stalls  += rhs.stalls;
        //     return *this;
        // }

        void add_rt_latency_dist(unsigned *update) {
            for (unsigned i=0; i<warp_statuses; i++) {
                for (unsigned j=0; j<ray_statuses; j++) {
                    rt_latency_dist[i][j] += update[i*ray_statuses + j];
                }
            }
            rt_latency_counter++;
        }
	};
  
    std::vector<SimPort<instr_trace_t*>> Inputs; 
    std::vector<SimPort<instr_trace_t*>> Outputs; 

    std::vector<std::vector<SimPort<MemReq>>> rtu_dcache_req_out;
    std::vector<std::vector<SimPort<MemRsp>>> rtu_dcache_rsp_in;
    RTUnit(const SimContext &ctx, const char* name, const Arch &arch, const DCRS &dcrs, Core* core);
    ~RTUnit();
    void reset();
    void tick();

    void get_csr(uint32_t addr, uint32_t wid, uint32_t tid, Word* value);
    void set_csr(uint32_t addr, uint32_t wid, uint32_t tid, Word value);

    void dcache_read(void* data, uint64_t addr, uint32_t size);
    void dcache_write(const void* data, uint64_t addr, uint32_t size);

    void traverse(uint32_t wid, std::vector<reg_data_t>& rd_data, RtuTraceData* trace_data);
    const PerfStats& perf_stats() const;
private:
	class Impl;
	Impl* impl_;
};

}