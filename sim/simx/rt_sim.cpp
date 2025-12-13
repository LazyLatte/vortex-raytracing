#include "rt_sim.h"
#include "rt_trace.h"
#include <set>
#include <deque>
#include <vector>


#define NUM_WARP_BUFFERS 8






class RTSim::Impl {
public:
    Impl(RTUnit* simobject, const Config& config)
        : simobject_(simobject)
        , core_(core)
        , arch_(arch)
        , dcrs_(dcrs)
        , perf_stats_()
        , num_blocks_(NUM_RTU_BLOCKS)
        , num_lanes_(NUM_RTU_LANES)
        , ray_buffers_(arch.num_warps())
        , pending_reqs_(MEM_QUEUE_SIZE),

        ,warp_buffers_(NUM_WARP_BUFFERS)
    {}

    ~Impl() {}

    void reset() {}

    void tick() {
    }

    const PerfStats& perf_stats() const {
        return perf_stats_;
    }

    unsigned process_returned_mem_access(const mem_fetch *mf) {
      // RT_DPRINTF("Processing returned data 0x%x (base 0x%x) for Warp %d\n", mf->get_uncoalesced_addr(), mf->get_uncoalesced_base_addr(), m_warp_id);
      // RT_DPRINTF("Current state:\n");
      // print_rt_accesses();
      // print_intersection_delay();
      // RT_DPRINTF("\n");
      
      // Count how many threads used this mf
      unsigned thread_found = 0;
      
      // Get addresses
      uint32_t addr = mf->get_addr();
      uint32_t uncoalesced_base_addr = mf->get_uncoalesced_base_addr();
      
      // Iterate through every thread in the warp
      for (unsigned i=0; i<num_lanes_; i++) {
        bool mem_record_done;
        if (process_returned_mem_access(mem_record_done, i, addr, uncoalesced_base_addr)) {
          thread_found++;
        }
      }
      return thread_found;
    }

    bool process_returned_mem_access(const mem_fetch *mf, unsigned tid) {
      bool mem_record_done = false;
      
      // Get addresses
      new_addr_type addr = mf->get_addr();
      new_addr_type uncoalesced_base_addr = mf->get_uncoalesced_base_addr();

      assert(process_returned_mem_access(mem_record_done, tid, addr, uncoalesced_base_addr));
      return mem_record_done;
    }

    bool process_returned_mem_access(bool &mem_record_done, unsigned tid, uint32_t addr, uint32_t uncoalesced_base_addr) {
      bool thread_found = false;
      if (!m_per_scalar_thread[tid].RT_mem_accesses.empty()) {
        RTMemoryTransactionRecord &mem_record = m_per_scalar_thread[tid].RT_mem_accesses.front();
        uint32_t thread_addr = mem_record.address;
        
        if (thread_addr == uncoalesced_base_addr) {
          new_addr_type coalesced_base_addr = line_size_based_tag_func(uncoalesced_base_addr, 32);
          unsigned position = (addr - coalesced_base_addr) / 32;
          std::string bitstring = mem_record.mem_chunks.to_string();
          RT_DPRINTF("Thread %d received chunk %d (of <%s>)\n", tid, position, bitstring.c_str());
          mem_record.mem_chunks.reset(position);
          
          // If all the bits are clear, the entire data has returned, pop from list
          if (mem_record.mem_chunks.none()) {
            // Set up delay of next intersection test
            unsigned n_delay_cycles = m_config->m_rt_intersection_latency.at(mem_record.type);
            m_per_scalar_thread[tid].intersection_delay += n_delay_cycles;
            
            RT_DPRINTF("Thread %d collected all chunks for address 0x%x (size %d)\n", tid, mem_record.address, mem_record.size);
            RT_DPRINTF("Processing data of transaction type %d for %d cycles.\n", mem_record.type, n_delay_cycles);
            m_per_scalar_thread[tid].RT_mem_accesses.pop_front();
            mem_record_done = true;

            // Mark triangle hit to store to memory
            if (mem_record.type == TransactionType::BVH_QUAD_LEAF_HIT) {
              m_per_scalar_thread[tid].ray_intersect = true;
              RT_DPRINTF("Buffer store detected for warp %d thread %d\n", m_uid, tid);
            }
          }
          thread_found = true;
        }
        
        // If the RT_mem_accesses is now empty, then the last memory request has returned and the thread is almost done
        if (m_per_scalar_thread[tid].RT_mem_accesses.empty()) {
          unsigned long long current_cycle =  GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle +
                                              GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
          m_per_scalar_thread[tid].end_cycle = current_cycle + m_per_scalar_thread[tid].intersection_delay;
        }
      }
      return thread_found;
    }


    void process_memory_response(instr_trace_t *trace) {         
      for (auto it=m_current_warps.begin(); it!=m_current_warps.end(); it++) {
        if (it->second.warp_id() == mf->get_wid()) {
          it->second.process_returned_mem_access(mf);
        }
      } 
    }

    void memory_cycle(SimPort<MemReq> &dcache_req_port) {

      auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);

      if (trace_data->rt_mem_accesses_empty()){
        return;
      }

      RTMemoryTransactionRecord next_access = trace_data->get_next_rt_mem_transaction();

      MemReq mem_req;
      mem_req.addr  = next_access.addr;
      mem_req.write = false;
      mem_req.tag   = tag;
      mem_req.cid   = trace->cid;
      mem_req.uuid  = trace->uuid;
      dcache_req_port.push(mem_req, 4);
      //DT(3, simobject_->name() << "-rt-req: addr=0x" << std::hex << mem_access.addr << ", tag=" << tag << ", tid=" << t << ", "<< trace);
      //++perf_stats_.reads;
    }


    void tick(){
      auto& input = simobject_->Inputs.at(iw);

      // Cycle intersection tests + get stats
      unsigned n_threads = 0;
      unsigned active_threads = 0;
      std::map<uint32_t, uint32_t> addr_set;
      for (auto it=m_current_warps.begin(); it!=m_current_warps.end(); ++it) {
        n_threads += (it->second).dec_thread_latency(mem_store_q);
        active_threads += (it->second).get_rt_active_threads();
        (it->second).num_unique_mem_access(addr_set);
      }

      // response
      for(uint32_t iw = 0; iw < num_blocks_; iw++){
          for (uint32_t t = 0; t < num_lanes_; t++) {
              auto& dcache_rsp_port = core_->rtu_dcache_rsp_in.at(iw).at(t);
              if (dcache_rsp_port.empty())
                  continue;
              auto& mem_rsp = dcache_rsp_port.front();
              auto& entry = pending_reqs_.at(mem_rsp.tag);
              process_memory_response(entry.trace);
              // auto trace = entry.trace;
              // DT(3, simobject_->name() << "-rt-rsp: tag=" << mem_rsp.tag << ", tid=" << t << ", " << *trace);
              // assert(entry.count);
              // --entry.count; // track remaining addresses
              // if (0 == entry.count) {
              //     auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);
              //     simobject_->Outputs.at(iw).push(trace, trace_data->pipeline_latency);
              //     pending_reqs_.release(mem_rsp.tag);
              // }
              dcache_rsp_port.pop();
          }
      }
      //writeback();

      // Change warp
      // if(...){
      //   input.push(trace, 1); //delay = 1
      //   input.pop();
      //   trace = input.front();
      // }

      // Schedule next memory request
      memory_cycle();

      // stage2: iterate into warp buffer in roundrobin

      // stage1: push a warp into wapr buf
      for (uint32_t iw = 0; iw  <num_blocks_; ++iw){
            auto& input = simobject_->Inputs.at(iw);

            if (input.empty())
                continue;

            if (warp_buffer.full()) {
                ++perf_stats_.stalls;
                return;
            }
            
            auto trace = input.front();
            warp_buffer.push(trace);
            input.pop();
        }
    }

private:
    RTUnit*       simobject_;
    PerfStats     perf_stats_;

    uint32_t num_blocks_;
    uint32_t num_lanes_;

    unsigned n_warps;

    struct warp_buffer_t{
      int t;
    };

    std::vector<warp_buffer_t> warp_buffers_;
    instr_trace_t *trace;
};

/* Start of RT unit functions */

// void warp_inst_t::print_rt_accesses() {
//   for (unsigned i=0; i<m_config->warp_size; i++) {
//     RT_DPRINTF("Thread %d: ", i);
//     for (auto it=m_per_scalar_thread[i].RT_mem_accesses.begin(); it!=m_per_scalar_thread[i].RT_mem_accesses.end(); it++) {
//       RT_DPRINTF("0x%x\t", it->address);
//     }
//     RT_DPRINTF("\n");
//   }
// }

// void warp_inst_t::print_intersection_delay() {
//   RT_DPRINTF("Intersection Delays: [");
//   for (unsigned i=0; i<m_config->warp_size; i++) {
//     RT_DPRINTF("%d\t", m_per_scalar_thread[i].intersection_delay);
//   }
//   RT_DPRINTF("\n");
// }

bool warp_inst_t::check_pending_writes(new_addr_type addr) {
  if (m_pending_writes.find(addr) != m_pending_writes.end()) {
    m_pending_writes.erase(addr);
    return true;
  }
  else {
    return false;
  }
}

unsigned warp_inst_t::dec_thread_latency(std::deque<std::pair<unsigned, new_addr_type> > &store_queue) { 
  // Track number of threads performing intersection tests
  unsigned n_threads = 0;
  
  for (unsigned i=0; i<m_config->warp_size; i++) {
    if (m_per_scalar_thread[i].intersection_delay > 0) {
      m_per_scalar_thread[i].intersection_delay--; 
      n_threads++;
      if (m_per_scalar_thread[i].intersection_delay == 0 && m_per_scalar_thread[i].ray_intersect) {
        // Temporary size
        unsigned size = RT_WRITE_BACK_SIZE;

        // Get an address to write to
        void* next_buffer_addr = GPGPUSim_Context(GPGPU_Context())->get_device()->get_gpgpu()->gpu_malloc(size);
        store_queue.push_back(std::pair<unsigned, new_addr_type>(m_uid, (new_addr_type)next_buffer_addr));
        m_per_scalar_thread[i].ray_intersect = false;
        RT_DPRINTF("Buffer store pushed for warp %d thread %d at 0x%x\n", m_uid, i, next_buffer_addr);

        m_pending_writes.insert((new_addr_type)next_buffer_addr);
      }
      
      for(auto & store_transaction : m_per_scalar_thread[i].RT_store_transactions) {
        store_queue.push_back(std::pair<unsigned, new_addr_type>(m_uid, (new_addr_type)(store_transaction.address)));
        RT_DPRINTF("Buffer store pushed for warp %d thread %d at 0x%x\n", m_uid, i, store_transaction.address);

        assert(m_pending_writes.find((new_addr_type)store_transaction.address) == m_pending_writes.end());
        m_pending_writes.insert((new_addr_type)store_transaction.address);
      }
      m_per_scalar_thread[i].RT_store_transactions.clear();
    }
  }
  
  return n_threads;
}

void warp_inst_t::track_rt_cycles(bool active) {
  bool stalled = is_stalled();
  unsigned warp_status = active ? warp_executing : stalled ? warp_stalled : warp_waiting;

  // Check progress of each thread
  for (unsigned i=0; i<m_config->warp_size; i++) {
    // Only check active threads
    if (thread_active(i)) {
      // Easiest check is intersection tests
      if (m_per_scalar_thread[i].intersection_delay != 0) {
        m_per_scalar_thread[i].status_num_cycles[warp_status][executing_op]++;
      }
      // Check that the thread is not done and not performing intersection tests. 
      else if (!m_per_scalar_thread[i].RT_mem_accesses.empty()) {
        // This is the next address that the thread wants
        RTMemoryTransactionRecord mem_record = m_per_scalar_thread[i].RT_mem_accesses.front();
        if (mem_record.status == RT_MEM_UNMARKED) {
          m_per_scalar_thread[i].status_num_cycles[warp_status][awaiting_scheduling]++;
        }
        else {
          m_per_scalar_thread[i].status_num_cycles[warp_status][awaiting_mf]++;
        }
      }
      // Otherwise the thread must be done or inactive
      else {
        m_per_scalar_thread[i].status_num_cycles[warp_status][trace_complete]++;
      }
    }
  }
}

unsigned * warp_inst_t::get_latency_dist(unsigned i) {
  return (unsigned *)m_per_scalar_thread[i].status_num_cycles;
}

// void warp_inst_t::set_rt_mem_transactions(unsigned int tid, std::vector<MemoryTransactionRecord> transactions) {
//   // Initialize
//   if (!m_per_scalar_thread_valid) {
//     m_per_scalar_thread.resize(m_config->warp_size);
//     m_per_scalar_thread_valid = true;
//   }
  
//   for (auto it=transactions.begin(); it!=transactions.end(); it++) {
//     // Convert transaction type and add to thread
//     RTMemoryTransactionRecord mem_record(
//       (new_addr_type)it->address,
//       it->size,
//       it->type
//     );
//     m_per_scalar_thread[tid].RT_mem_accesses.push_back(mem_record);
//   }
// }

// void warp_inst_t::set_rt_mem_store_transactions(unsigned int tid, std::vector<MemoryStoreTransactionRecord>& transactions) {
//   m_per_scalar_thread[tid].RT_store_transactions = transactions;
// }

bool warp_inst_t::is_stalled() {
  // If there are still memory requests waiting to be processed, not stalled
  if (!m_next_rt_accesses_set.empty()) {
    return false;
  }
  // Otherwise check every thread
  for (unsigned i=0; i<m_config->warp_size; i++) {
    if (!m_per_scalar_thread[i].RT_mem_accesses.empty()) {
      RTMemoryTransactionRecord mem_record = m_per_scalar_thread[i].RT_mem_accesses.front();
      
      // If there is an unprocessed record, not stalled
      if (mem_record.status == RT_MEM_UNMARKED && m_per_scalar_thread[i].intersection_delay == 0) return false;
    }
  }
  
  // Otherwise stalled
  return true;
}

void warp_inst_t::set_rt_ray_properties(unsigned int tid, Ray ray) {
  assert(m_per_scalar_thread_valid);
  m_per_scalar_thread[tid].ray_properties = ray;
}

bool warp_inst_t::rt_mem_accesses_empty() { 
  bool empty = true;
  for (unsigned i = 0; i < m_config->warp_size; i++) {
    empty &= m_per_scalar_thread[i].RT_mem_accesses.empty();
  }
  empty &= m_next_rt_accesses_set.empty();
  return empty;
}

void warp_inst_t::num_unique_mem_access(std::map<new_addr_type, unsigned> &addr_set) {
  for (unsigned i = 0; i < m_config->warp_size; i++) {
    if (!m_per_scalar_thread[i].RT_mem_accesses.empty()) {
      RTMemoryTransactionRecord record = m_per_scalar_thread[i].RT_mem_accesses.front();
      addr_set[record.address]++;
    }
  }
}

bool warp_inst_t::rt_intersection_delay_done() { 
  bool done = true;
  for (unsigned i = 0; i < m_config->warp_size; i++) {
    done &= (m_per_scalar_thread[i].intersection_delay == 0);
  }
  return done;
}

unsigned warp_inst_t::get_rt_active_threads() {
  assert(m_per_scalar_thread_valid);
  unsigned active_threads = 0;
  for (auto it=m_per_scalar_thread.begin(); it!=m_per_scalar_thread.end(); it++) {
    if (!it->RT_mem_accesses.empty()) {
      active_threads++;
    }
  }
  return active_threads;
}

std::deque<unsigned> warp_inst_t::get_rt_active_thread_list() {
  assert(m_per_scalar_thread_valid);
  std::deque<unsigned> active_threads;
  for (unsigned i=0; i<m_config->warp_size; i++) {
    if (!m_per_scalar_thread[i].RT_mem_accesses.empty()) {
      active_threads.push_back(i);
    }
  }
  return active_threads;
}

void warp_inst_t::set_thread_end_cycle(unsigned long long cycle) {
    for (unsigned i=0; i<m_config->warp_size; i++) {
        m_per_scalar_thread[i].end_cycle = cycle;
    }
}

void warp_inst_t::update_next_rt_accesses() {
  
  // Iterate through every thread
  for (unsigned i=0; i<num_lanes_; i++) {
    if (!m_per_scalar_thread[i].RT_mem_accesses.empty()) {
      RTMemoryTransactionRecord next_access = m_per_scalar_thread[i].RT_mem_accesses.front();
      
      // If "unmarked", this has not been added to queue yet (also make sure intersection is complete)
      if (next_access.status == RTMemStatus::RT_MEM_UNMARKED && m_per_scalar_thread[i].intersection_delay == 0) {
        std::pair<new_addr_type, unsigned> address_size_pair (next_access.address, next_access.size);
        // Add to queue if the same address doesn't already exist
        if (m_next_rt_accesses_set.find(address_size_pair) == m_next_rt_accesses_set.end()) {
          m_next_rt_accesses.push_back(next_access);
          m_next_rt_accesses_set.insert(address_size_pair);
        }
        // Update status
        m_per_scalar_thread[i].RT_mem_accesses.front().status = RTMemStatus::RT_MEM_AWAITING;
      }
    }
  }
  
}

RTMemoryTransactionRecord warp_inst_t::get_next_rt_mem_transaction() {
  // Update the list of next accesses
  update_next_rt_accesses();
  RTMemoryTransactionRecord next_access;
  std::pair<new_addr_type, unsigned> address_size_pair;
  
  do {
    // Choose the next one on the list
    next_access = m_next_rt_accesses.front();
    m_next_rt_accesses.pop_front();
    
    address_size_pair = std::pair<new_addr_type, unsigned>(next_access.address, next_access.size);
    
  // Check that the address hasn't already been sent
  } while (m_next_rt_accesses_set.find(address_size_pair) == m_next_rt_accesses_set.end());
  
  m_next_rt_accesses_set.erase(address_size_pair);
  
  RT_DPRINTF("Next access chosen: 0x%x with %dB\n", next_access.address, next_access.size);
  return next_access;
}

void warp_inst_t::undo_rt_access(new_addr_type addr){ 
  std::pair<new_addr_type, unsigned> address_size_pair (addr, 32);
  assert (m_next_rt_accesses_set.find(address_size_pair) == m_next_rt_accesses_set.end());
  
  // Repackage address into a transaction record
  RTMemoryTransactionRecord mem_record(
    addr, 32, TransactionType::UNDEFINED
  );
  
  // Already in queue
  mem_record.status = RT_MEM_AWAITING;

  m_next_rt_accesses.push_front(mem_record);
  m_next_rt_accesses_set.insert(address_size_pair);
  RT_DPRINTF("UNDO: 0x%x added back to queue\n", addr);
}

unsigned warp_inst_t::process_returned_mem_access(const mem_fetch *mf) {
  RT_DPRINTF("Processing returned data 0x%x (base 0x%x) for Warp %d\n", mf->get_uncoalesced_addr(), mf->get_uncoalesced_base_addr(), m_warp_id);
  RT_DPRINTF("Current state:\n");
  print_rt_accesses();
  print_intersection_delay();
  RT_DPRINTF("\n");
  
  // Count how many threads used this mf
  unsigned thread_found = 0;
  
  // Get addresses
  new_addr_type addr = mf->get_addr();
  new_addr_type uncoalesced_base_addr = mf->get_uncoalesced_base_addr();
  
  // Iterate through every thread in the warp
  for (unsigned i=0; i<m_config->warp_size; i++) {
    bool mem_record_done;
    if (process_returned_mem_access(mem_record_done, i, addr, uncoalesced_base_addr)) {
      thread_found++;
    }
  }
  return thread_found;
}

bool warp_inst_t::process_returned_mem_access(const mem_fetch *mf, unsigned tid) {
  bool mem_record_done = false;
  
  // Get addresses
  new_addr_type addr = mf->get_addr();
  new_addr_type uncoalesced_base_addr = mf->get_uncoalesced_base_addr();

  assert(process_returned_mem_access(mem_record_done, tid, addr, uncoalesced_base_addr));
  return mem_record_done;
}

bool warp_inst_t::process_returned_mem_access(bool &mem_record_done, unsigned tid, new_addr_type addr, new_addr_type uncoalesced_base_addr) {
  bool thread_found = false;
  if (!m_per_scalar_thread[tid].RT_mem_accesses.empty()) {
    RTMemoryTransactionRecord &mem_record = m_per_scalar_thread[tid].RT_mem_accesses.front();
    new_addr_type thread_addr = mem_record.address;
    
    if (thread_addr == uncoalesced_base_addr) {
      new_addr_type coalesced_base_addr = line_size_based_tag_func(uncoalesced_base_addr, 32);
      unsigned position = (addr - coalesced_base_addr) / 32;
      std::string bitstring = mem_record.mem_chunks.to_string();
      RT_DPRINTF("Thread %d received chunk %d (of <%s>)\n", tid, position, bitstring.c_str());
      mem_record.mem_chunks.reset(position);
      
      // If all the bits are clear, the entire data has returned, pop from list
      if (mem_record.mem_chunks.none()) {
        // Set up delay of next intersection test
        unsigned n_delay_cycles = m_config->m_rt_intersection_latency.at(mem_record.type);
        m_per_scalar_thread[tid].intersection_delay += n_delay_cycles;
        
        RT_DPRINTF("Thread %d collected all chunks for address 0x%x (size %d)\n", tid, mem_record.address, mem_record.size);
        RT_DPRINTF("Processing data of transaction type %d for %d cycles.\n", mem_record.type, n_delay_cycles);
        m_per_scalar_thread[tid].RT_mem_accesses.pop_front();
        mem_record_done = true;

        // Mark triangle hit to store to memory
        if (mem_record.type == TransactionType::BVH_QUAD_LEAF_HIT) {
          m_per_scalar_thread[tid].ray_intersect = true;
          RT_DPRINTF("Buffer store detected for warp %d thread %d\n", m_uid, tid);
        }
      }
      thread_found = true;
    }
    
    // If the RT_mem_accesses is now empty, then the last memory request has returned and the thread is almost done
    if (m_per_scalar_thread[tid].RT_mem_accesses.empty()) {
      unsigned long long current_cycle =  GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle +
                                          GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
      m_per_scalar_thread[tid].end_cycle = current_cycle + m_per_scalar_thread[tid].intersection_delay;
    }
  }
  return thread_found;
}