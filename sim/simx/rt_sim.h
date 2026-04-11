#pragma once
#include "types.h"
#include "rt_unit.h"
#include "rt_trace.h"
#include <unordered_map>
#include <unordered_set>

namespace vortex {
	
class RTSim {
public:
	RTSim(RTUnit* simobject);
	~RTSim(){};

	void reset();
	void tick();
	const RTUnit::PerfStats& perf_stats() const { return perf_stats_; }
private:
	bool process_memory_request();
	bool process_memory_response();
	void process_memory_response(instr_trace_t *rsp_trace, uint32_t rsp_addr);
	bool process_intersection_delay();
	void schedule_warp();
	void add_warp();
	void remove_warp(instr_trace_t *target_trace);
	void check_completion();

    uint32_t num_blocks_;
    uint32_t num_lanes_;

	struct pending_req_t {
		instr_trace_t* trace;
		uint32_t addr;
	};

	HashTable<pending_req_t> pending_reqs_;

	std::deque<std::pair<instr_trace_t*, MemoryStoreTransactionRecord>> mem_store_q;
	std::unordered_map<instr_trace_t*, unsigned long long> warp_latencies_;
	std::unordered_map<instr_trace_t*, uint32_t> warp_iws_;
    std::unordered_set<instr_trace_t*> warp_buffers_;
	RTUnit::PerfStats perf_stats_;
	instr_trace_t* cur_trace;

	RTUnit*  simobject_;
};

}