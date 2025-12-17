#pragma once
#include "types.h"
#include "rt_unit.h"
#include "rt_trace.h"

namespace vortex {
	
class RTSim {
public:
	RTSim(RTUnit* simobject/*, const Config& config*/);
	~RTSim();

	void reset();
	void tick();

private:
	void process_memory_request(instr_trace_t *trace);
	void process_memory_response();
	void process_memory_response(instr_trace_t *rsp_trace, uint32_t rsp_addr);
	void process_intersection_delay();
	void schedule_warp(instr_trace_t **scheduled_trace);
	void add_warp();
	void remove_warp(instr_trace_t *target_trace);
	void check_completion();


    RTUnit*  simobject_;

    uint32_t num_blocks_;
    uint32_t num_lanes_;

	struct pending_req_t {
		instr_trace_t* trace;
		uint32_t addr;
	};

	HashTable<pending_req_t> pending_reqs_;

	std::deque<std::pair<uint32_t, uint32_t> > mem_store_q;

    std::vector<instr_trace_t*> warp_buffers_;
};

}