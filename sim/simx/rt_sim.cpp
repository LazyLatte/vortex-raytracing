#include "rt_sim.h"

#include <set>
#include <deque>
#include <vector>
#include <iostream>

#define MEM_QUEUE_SIZE 256
#define MAX_NUM_WARP_BUFFERS 8

using namespace vortex;

RTSim::RTSim(RTUnit* simobject/*, const Config& config*/)
  : simobject_(simobject)
  , num_blocks_(NUM_RTU_BLOCKS)
  , num_lanes_(NUM_RTU_LANES)
  //, warp_buffers_(MAX_NUM_WARP_BUFFERS)
  , pending_reqs_(MEM_QUEUE_SIZE)
{}

RTSim::~RTSim(){

}

void RTSim::add_warp(){
  if (warp_buffers_.size() >= MAX_NUM_WARP_BUFFERS) {
      //++perf_stats_.stalls;
      return;
  }
  
  for (uint32_t iw = 0; iw < num_blocks_; ++iw){
    auto& input = simobject_->Inputs.at(iw);

    if (!input.empty()){
      auto trace = input.front();
      warp_buffers_.push_back(trace);
      input.pop();
      return;
    }
  }
}

void RTSim::remove_warp(instr_trace_t *target_trace){
  auto it = std::find_if(warp_buffers_.begin(), warp_buffers_.end(), [&](const instr_trace_t* trace){ 
    return trace->wid == target_trace->wid; 
  });

  if (it != warp_buffers_.end())
    warp_buffers_.erase(it);
  else
    std::cout << "Cannot remove absent warp!!" << std::endl;
}

void RTSim::schedule_warp(instr_trace_t **scheduled_trace){
  if (mem_store_q.empty()) {
    // Choose next warp

    // Return if there are no warps in the RT unit
    if (warp_buffers_.empty()){
      *scheduled_trace = nullptr;
      return;
    }
        
    // Otherwise, find the first non-stalled warp
    else {
      for (instr_trace_t *trace : warp_buffers_) {
        auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
        if (!trace_data->is_stalled()) { 
          *scheduled_trace = trace;
          break;
        }
      }
      // if (!scheduled_trace->empty()){
      //   remove_warp(scheduled_trace);
      // }
    }
  }
  // Get cycle status
  // if (!rt_inst.empty()) rt_inst.track_rt_cycles(true);
  // for (auto it=m_current_warps.begin(); it!=m_current_warps.end(); it++) {
  //   (it->second).track_rt_cycles(false);
  // }
}

void RTSim::process_intersection_delay(){
  unsigned n_threads = 0;
  unsigned active_threads = 0;
  std::map<uint32_t, uint32_t> addr_set;
  for (instr_trace_t *trace : warp_buffers_) {
    auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
    n_threads += trace_data->dec_thread_latency(mem_store_q);
    active_threads += trace_data->get_rt_active_threads();
    trace_data->num_unique_mem_access(addr_set);
  }
}

void RTSim::process_memory_response(instr_trace_t *rsp_trace, uint32_t rsp_addr){
  for (instr_trace_t *trace : warp_buffers_) {
    if (trace->wid == rsp_trace->wid) {
      auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
      uint32_t thread_found = trace_data->process_returned_mem_access(rsp_addr);
    }
  } 
}

void RTSim::process_memory_response(){
  for(uint32_t iw = 0; iw < num_blocks_; iw++){
    for (uint32_t t = 0; t < num_lanes_; t++) {
      auto& dcache_rsp_port = simobject_->rtu_dcache_rsp_in.at(iw).at(t);
      if (dcache_rsp_port.empty())
          continue;
      auto& mem_rsp = dcache_rsp_port.front();
      auto& entry = pending_reqs_.at(mem_rsp.tag);
      auto rsp_trace = entry.trace;
      auto rsp_addr = entry.addr;
      process_memory_response(rsp_trace, rsp_addr);

      pending_reqs_.release(mem_rsp.tag);
      dcache_rsp_port.pop();
    }
  }
}

void RTSim::process_memory_request(instr_trace_t *trace){
  if(trace == nullptr) return;
  //if (!inst.active_count()) return;
  
  auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
  if (trace_data->is_stalled() || trace_data->rt_mem_accesses_empty()){
    return;
  }

  RTMemoryTransactionRecord next_access = trace_data->get_next_rt_mem_transaction();

  auto tag = pending_reqs_.allocate({trace, next_access.addr});

  MemReq mem_req;
  mem_req.addr  = next_access.addr;
  mem_req.write = false;
  mem_req.tag   = tag;
  mem_req.cid   = trace->cid;
  mem_req.uuid  = trace->uuid;

  auto& dcache_req_port = simobject_->rtu_dcache_req_out.at(0).at(0); //???
  dcache_req_port.push(mem_req, 1);
}

void RTSim::check_completion(){
  // Check to see if any warps are complete
  //instr_trace_t *completed_trace = nullptr;
  for (instr_trace_t *trace : warp_buffers_) {
    auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
   // RT_DPRINTF("Checking warp inst uid: %d\n", debug_inst.get_uid());
    // A completed warp has no more memory accesses and all the intersection delays are complete and has no pending writes
    if (trace_data->rt_mem_accesses_empty() && trace_data->rt_intersection_delay_done() && !trace_data->has_pending_writes()) {
      //RT_DPRINTF("Shader %d: Warp %d (uid: %d) completed!\n", m_sid, it->second.warp_id(), it->first);
      //completed_trace = trace;
      // n_warps--;
      // assert(n_warps >= 0 && n_warps <= MAX_NUM_WARP_BUFFERS);
      simobject_->Outputs.at(0).push(trace, 1);
      remove_warp(trace);
    }else{
      //RT_DPRINTF("Cycle: %d, Warp inst uid: %d not done. rt_mem_accesses_empty: %d, rt_intersection_delay_done: %d, no pending_writes: %d\n", GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle, debug_inst.get_uid(), it->second.rt_mem_accesses_empty(), it->second.rt_intersection_delay_done(), !it->second.has_pending_writes());
    }
  }
  
  // Remove complete warp
  // if (completed_trace != nullptr) {
  //   warp_buffers_.erase(completed_trace);
  // }
  
  //assert(n_warps == warp_buffers_.size());
}

void RTSim::tick(){
  process_intersection_delay();
  process_memory_response();
  //writeback()
  add_warp();

  instr_trace_t *trace;
  schedule_warp(&trace);
  process_memory_request(trace);

  check_completion();
}

void RTSim::reset(){
  pending_reqs_.clear();
}