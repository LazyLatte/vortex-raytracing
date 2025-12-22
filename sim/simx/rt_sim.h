#pragma once
#include "types.h"
#include "rt_unit.h"
#include "rt_trace.h"
#include <unordered_map>
namespace vortex {
	
class RTSim {
public:
	RTSim(RTUnit* simobject/*, const Config& config*/);
	~RTSim();

	void reset();
	void tick();
	const RTUnit::PerfStats& perf_stats() const { return perf_stats_; }
private:
	void process_memory_request();
	void process_memory_response();
	void process_memory_response(instr_trace_t *rsp_trace, uint32_t rsp_addr);
	void process_intersection_delay();
	void schedule_warp();
	void add_warp();
	void remove_warp(instr_trace_t *target_trace);
	void check_completion();
	void cycle();

	instr_trace_t *get_warp_trace(int wid);
    RTUnit*  simobject_;

    uint32_t num_blocks_;
    uint32_t num_lanes_;

	struct pending_req_t {
		instr_trace_t* trace;
		uint32_t addr;
	};

	HashTable<pending_req_t> pending_reqs_;

	std::deque<std::pair<uint32_t, uint32_t>> mem_store_q;
	std::unordered_map<uint32_t, unsigned long long> warp_latencies_;
	
    std::vector<instr_trace_t*> warp_buffers_;
	RTUnit::PerfStats perf_stats_;

	int current_wid;
};

}