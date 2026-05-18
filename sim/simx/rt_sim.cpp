#include "rt_sim.h"
#include <set>
#include <deque>
#include <vector>
#include <iostream>
#include <cassert>
#define MEM_QUEUE_SIZE 2048

using namespace vortex;

RTSim::RTSim(RTUnit* simobject)
  : simobject_(simobject)
  , num_blocks_(NUM_RTU_BLOCKS)
  , num_lanes_(NUM_RTU_LANES)
  , pending_reqs_(MEM_QUEUE_SIZE)
  , cur_trace(nullptr)
{}

void RTSim::add_warp(){
  if (warp_buffers_.size() >= RT_WARP_BUFFER_SIZE) return;

  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw){
    auto& input = simobject_->Inputs.at(iw);
		if (input.empty())
			continue;

    auto trace = input.peek();
    auto op_type = std::get<RtuType>(trace->op_type);

    if(op_type != RtuType::TRACE && op_type != RtuType::COMMIT){
      if(simobject_->Outputs.at(iw).try_send(trace, 1)){
        input.pop();
      }
      continue;
    }

    auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
    if(trace_data->rt_mem_accesses_empty()){
      if(simobject_->Outputs.at(iw).try_send(trace, 1)){
        input.pop();
      }
      continue;
    }

    assert(warp_buffers_.count(trace) == 0);
    warp_buffers_.insert(trace);
    warp_latencies_[trace] = 0;
    warp_iws_[trace] = iw;
    input.pop();
    break;
  }
}

void RTSim::remove_warp(instr_trace_t *trace){
  assert(warp_buffers_.count(trace));
  warp_buffers_.erase(trace);
}

void RTSim::schedule_warp(){
  if (mem_store_q.empty()) {
    // Choose next warp

    // Return if there are no warps in the RT unit
    if (warp_buffers_.empty()){
      cur_trace = nullptr;
      return;
    }
        
    // Otherwise, find the first non-stalled warp
    else {
      cur_trace = nullptr;
      for (instr_trace_t *trace : warp_buffers_) {
        auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
        if (!trace_data->is_stalled()) { 
          cur_trace = trace;
          break;
        }
      }
    }
  }
  // Update cycle status
  for (instr_trace_t *trace : warp_buffers_) {
    auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
    trace_data->track_rt_cycles(trace == cur_trace, trace->tmask);
  }
}

bool RTSim::process_intersection_delay(){
  unsigned n_threads = 0;
  unsigned active_threads = 0;
  std::unordered_map<uint32_t, uint32_t> addr_set;
  for (instr_trace_t *trace : warp_buffers_) {
    auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
    n_threads += trace_data->dec_thread_latency(trace, mem_store_q);
    //active_threads += trace_data->get_rt_active_threads();
    //trace_data->num_unique_mem_access(addr_set);
  }
  return n_threads > 0;
}

void RTSim::process_memory_response(instr_trace_t *rsp_trace, uint32_t rsp_addr){
  for (instr_trace_t *trace : warp_buffers_) {
    if (trace == rsp_trace) {
      auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
      uint32_t thread_found = trace_data->process_returned_mem_access(rsp_addr);
    }
  } 
}

bool RTSim::process_memory_response(){
  auto& mem_rsp_port = simobject_->rtu_mem_rsp.at(0); // Block 0
  if (!mem_rsp_port.empty()){
    auto& mem_rsp = mem_rsp_port.peek();
    auto& entry = pending_reqs_.at(mem_rsp.tag);
    auto rsp_trace = entry.trace;
    auto rsp_addr = entry.addr;
    process_memory_response(rsp_trace, rsp_addr);
    pending_reqs_.release(mem_rsp.tag);
    mem_rsp_port.pop();

    return true;
  } 
  return false;
}

bool RTSim::process_memory_request(){

  auto& mem_req_port = simobject_->rtu_mem_req.at(0); // Block 0

  if(mem_req_port.full()) return false;

  if (!mem_store_q.empty()) {
    auto next_store = mem_store_q.front();
    mem_store_q.pop_front();

    auto trace = next_store.first;
    auto store_transaction = next_store.second;

    RtuReq mem_req;
    mem_req.addr  = store_transaction.addr;
    mem_req.size  = store_transaction.size;
    mem_req.write = true;
    mem_req.tag   = 0;
    mem_req.cid   = trace->cid;
    mem_req.uuid  = trace->uuid;

    mem_req_port.send(mem_req, 1);

    auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
    trace_data->m_pending_writes.erase(store_transaction.addr);
    return true;
  }

  auto trace = cur_trace;
  if(trace == nullptr) return false;  
  auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
  
  if(!trace_data->is_stalled() && !trace_data->rt_mem_accesses_empty()){    
    RTMemoryTransactionRecord next_access = trace_data->get_next_rt_mem_transaction();

    auto tag = pending_reqs_.allocate({trace, next_access.addr});

    RtuReq mem_req;
    mem_req.addr  = next_access.addr;
    mem_req.size  = next_access.size;
    mem_req.write = false;
    mem_req.tag   = tag;
    mem_req.cid   = trace->cid;
    mem_req.uuid  = trace->uuid;

    mem_req_port.send(mem_req, 1);
    return true;
  }

  return false;
}

void RTSim::check_completion(){
  // Check to see if any warps are complete
  instr_trace_t *target_trace = nullptr;  
  for (instr_trace_t *trace : warp_buffers_) {
    auto& output = simobject_->Outputs.at(warp_iws_[trace]);
    if(output.full()) continue;

    auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
    
    // A completed warp has no more memory accesses and all the intersection delays are complete and has no pending writes
    if (trace_data->rt_mem_accesses_empty() && trace_data->rt_intersection_delay_done() && !trace_data->has_pending_writes()) {      
      perf_stats_.rt_total_warps++;
      perf_stats_.rt_total_warp_latency += warp_latencies_[trace];
      perf_stats_.rt_warp_latencies.push_back(warp_latencies_[trace]);

      unsigned long long total_thread_cycles = 0;
      for (unsigned i=0; i<trace_data->m_per_scalar_thread.size(); i++) {
        if (trace->tmask.test(i)) {
          unsigned *latency_dist = trace_data->get_latency_dist(i);
          
          unsigned long long thread_cycles = 1; // trace complete takes 1 cycle
          for (unsigned j = 0; j < warp_statuses; ++j) {
              for (unsigned k = 0; k < ray_statuses; ++k) {
                  if (k != rt_ray_status::trace_complete) {
                    thread_cycles += latency_dist[j * ray_statuses + k];
                  }
              }
          }
          total_thread_cycles += thread_cycles;
          perf_stats_.add_rt_latency_dist(latency_dist);
        }
      }

      float avg_thread_cycles = (float)total_thread_cycles / trace_data->m_per_scalar_thread.size();
      perf_stats_.rt_total_thread_latency += avg_thread_cycles;

      float rt_simt_efficiency = (float)total_thread_cycles / (trace_data->m_per_scalar_thread.size() * warp_latencies_[trace]);
      perf_stats_.rt_total_simt_efficiency += rt_simt_efficiency;

      output.send(trace, warp_latencies_[trace]);
      warp_latencies_.erase(trace);
      warp_iws_.erase(trace);
      target_trace = trace;
      break;
    }
  }

  if(target_trace) remove_warp(target_trace);
}

void RTSim::tick(){
  check_completion();
  bool op_active = process_intersection_delay();
  bool rsp_active = process_memory_response();  
  bool req_active = process_memory_request();
  schedule_warp();
  add_warp();
  
  // update counters
  for (auto it = warp_latencies_.begin(); it != warp_latencies_.end(); ++it){
    it->second++;
  }

  if(op_active || rsp_active || req_active){
    perf_stats_.rt_active_cycles++;
  }

  if(warp_buffers_.size() > 0){
    perf_stats_.total_elapsed_cycles++;
  }
  
}

void RTSim::reset(){
  pending_reqs_.clear();
  perf_stats_ = RTUnit::PerfStats();
}