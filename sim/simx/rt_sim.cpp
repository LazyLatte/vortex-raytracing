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
  , pending_reqs_(MEM_QUEUE_SIZE)
  , current_wid(-1)
{}

RTSim::~RTSim(){}

void RTSim::add_warp(){
  if (warp_buffers_.size() >= MAX_NUM_WARP_BUFFERS) {
      return;
  }
  
  for (uint32_t iw = 0; iw < num_blocks_; ++iw){
    auto& input = simobject_->Inputs.at(iw);

    if (!input.empty()){
      auto trace = input.front();
      assert(trace != nullptr);
      auto op_type = std::get<RtuType>(trace->op_type);
      if(1){
        simobject_->Outputs.at(0).push(trace, 1);
        input.pop();
        continue;
      }

      warp_buffers_.push_back(trace);
      warp_latencies_.insert({trace->wid, 0});
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
    std::cout << "Cannot remove non-existent warp!!" << std::endl;
}

instr_trace_t* RTSim::get_warp_trace(int wid){
  for(instr_trace_t *trace : warp_buffers_){
    if(trace->wid == wid){
      return trace;
    }
  }
  return nullptr;
}

void RTSim::schedule_warp(){
  if (mem_store_q.empty()) {
    // Choose next warp

    // Return if there are no warps in the RT unit
    if (warp_buffers_.empty()){
      current_wid = -1;
      return;
    }
        
    // Otherwise, find the first non-stalled warp
    else {
      for (instr_trace_t *trace : warp_buffers_) {
        auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
        if (!trace_data->is_stalled()) { 
          current_wid = trace->wid;
          break;
        }
      }
    }
  }
  // Update cycle status
  for (instr_trace_t *trace : warp_buffers_) {
    auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
    trace_data->track_rt_cycles(trace->wid == current_wid, trace->tmask);
  }
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

void RTSim::process_memory_request(){
  auto trace = get_warp_trace(current_wid);

  if(trace == nullptr) return;
  //if (!inst.active_count()) return;
  
  auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
  if (trace_data->is_stalled() || trace_data->rt_mem_accesses_empty()){
    return;
  }

  auto& dcache_req_port = simobject_->rtu_dcache_req_out.at(0).at(0); //???

  if (!mem_store_q.empty()) {
    auto next_store = mem_store_q.front();
    uint32_t next_addr = next_store.second;
    uint32_t warp_uid = next_store.first;
    auto tag = pending_reqs_.allocate({trace, next_addr});

    MemReq mem_req;
    mem_req.addr  = next_addr;
    mem_req.write = true;
    mem_req.tag   = tag;
    mem_req.cid   = trace->cid; //fix
    mem_req.uuid  = trace->uuid; //fix

    dcache_req_port.push(mem_req, 1);
    mem_store_q.pop_front();
  }else{
    RTMemoryTransactionRecord next_access = trace_data->get_next_rt_mem_transaction();

    auto tag = pending_reqs_.allocate({trace, next_access.addr});

    MemReq mem_req;
    mem_req.addr  = next_access.addr;
    mem_req.write = false;
    mem_req.tag   = tag;
    mem_req.cid   = trace->cid;
    mem_req.uuid  = trace->uuid;

    dcache_req_port.push(mem_req, 1);
  }
}

void RTSim::check_completion(){
  // Check to see if any warps are complete
  for (instr_trace_t *trace : warp_buffers_) {
    auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);

    // A completed warp has no more memory accesses and all the intersection delays are complete and has no pending writes
    if (trace_data->rt_mem_accesses_empty() && trace_data->rt_intersection_delay_done() && !trace_data->has_pending_writes()) {
      perf_stats_.rt_total_warps++;
      perf_stats_.rt_total_warp_latency += warp_latencies_[trace->wid];

      unsigned long long total_thread_cycles = 0;
      for (unsigned i=0; i<trace_data->m_per_scalar_thread.size(); i++) {
        if (trace->tmask.test(i)) {
          unsigned *latency_dist = trace_data->get_latency_dist(i);
          
          unsigned long long thread_cycles = 1; // trace complete takes 1 cycle
          for (unsigned i=0; i<warp_statuses; i++) {
              for (unsigned j=0; j<ray_statuses; j++) {
                  if(j!=trace_complete){
                    thread_cycles += latency_dist[i*ray_statuses + j];
                  }
              }
          }
          total_thread_cycles += thread_cycles;
          perf_stats_.add_rt_latency_dist(latency_dist);
          //std::cout << thread_cycles << " ";
        }
      }
      //std::cout << ": " << warp_latencies_[trace->wid] << std::endl;

      float avg_thread_cycles = (float)total_thread_cycles / trace_data->m_per_scalar_thread.size();
      perf_stats_.rt_total_thread_latency += avg_thread_cycles;

      float rt_simt_efficiency = (float)total_thread_cycles / (trace_data->m_per_scalar_thread.size() * warp_latencies_[trace->wid]);
      perf_stats_.rt_total_simt_efficiency += rt_simt_efficiency;
      
      //std::cout << rt_simt_efficiency << std::endl;
      simobject_->Outputs.at(0).push(trace, warp_latencies_[trace->wid]);
      warp_latencies_.erase(trace->wid);
      remove_warp(trace);
    }
  }
}

void RTSim::cycle(){
  for (auto it = warp_latencies_.begin(); it != warp_latencies_.end(); ++it){
    it->second++;
  }
}

void RTSim::tick(){
  process_intersection_delay();
  process_memory_response();
  //writeback()
  
  add_warp();
  schedule_warp();
  cycle(); // latency++

  process_memory_request();
  check_completion();
}

void RTSim::reset(){
  pending_reqs_.clear();
  perf_stats_ = RTUnit::PerfStats();
}