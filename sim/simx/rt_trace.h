#pragma once
#include "instr_trace.h"
#include <set>
#include <map>

namespace vortex {

enum rt_warp_status {
  warp_stalled = 0,
  warp_waiting,
  warp_executing,
  warp_statuses
}; 

enum rt_ray_status {
  awaiting_processing = 0,
  awaiting_scheduling,
  awaiting_mf,
  executing_op,
  trace_complete,
  ray_statuses
};

enum RTMemStatus {
  RT_MEM_UNMARKED,
  RT_MEM_AWAITING,
  RT_MEM_COMPLETE,
};

enum class TransactionType {
    BVH_STRUCTURE,
    BVH_INTERNAL_NODE,
    BVH_INSTANCE_LEAF,
    BVH_PRIMITIVE_LEAF_DESCRIPTOR,
    BVH_QUAD_LEAF,
    BVH_QUAD_LEAF_HIT,
    BVH_PROCEDURAL_LEAF,
    Intersection_Table_Load,
    UNDEFINED,
};

enum class StoreTransactionType {Intersection_Table_Store, Traversal_Results};

struct RTMemoryTransactionRecord {
    uint32_t addr;
    uint32_t size;
    TransactionType type;
    RTMemStatus status;
    RTMemoryTransactionRecord(){}
    RTMemoryTransactionRecord(uint32_t addr, uint32_t size, TransactionType type)
    : addr(addr), size(size), type(type), status(RT_MEM_UNMARKED){}
};

struct MemoryStoreTransactionRecord {
    void* address;
    uint32_t size;
    StoreTransactionType type;

    MemoryStoreTransactionRecord(void* address, uint32_t size, StoreTransactionType type)
    : address(address), size(size), type(type) {}

};

struct per_thread_info {
  per_thread_info(){}
  ~per_thread_info(){}
                                                  
  // RT variables    
  std::deque<RTMemoryTransactionRecord> RT_mem_accesses;
  std::vector<MemoryStoreTransactionRecord> RT_store_transactions;
  bool ray_intersect = false;
  //Ray ray_properties;
  unsigned intersection_delay;
  unsigned status_num_cycles[warp_statuses][ray_statuses] = {};

  void clear_mem_accesses() {
    RT_mem_accesses.clear();
  }
};

//warp_instr_t
struct RtuTraceData : public ITraceData {
  using Ptr = std::shared_ptr<RtuTraceData>;
  std::vector<per_thread_info> m_per_scalar_thread;

  std::deque<RTMemoryTransactionRecord> m_next_rt_accesses;
  std::set<std::pair<uint32_t, uint32_t> > m_next_rt_accesses_set;
  std::set<uint32_t> m_pending_writes;
  RtuTraceData(uint32_t num_threads = 0) : m_per_scalar_thread(num_threads) {}

  bool has_pending_writes() { return !m_pending_writes.empty(); }
  bool rt_mem_accesses_empty(){
    bool empty = true;
    for (unsigned i = 0; i < m_per_scalar_thread.size(); i++) {
      empty &= m_per_scalar_thread[i].RT_mem_accesses.empty();
    }
    empty &= m_next_rt_accesses_set.empty();
    return empty;
  }

  void num_unique_mem_access(std::map<uint32_t, uint32_t> &addr_set) {
    for (unsigned i = 0; i < m_per_scalar_thread.size(); i++) {
      if (!m_per_scalar_thread[i].RT_mem_accesses.empty()) {
        RTMemoryTransactionRecord record = m_per_scalar_thread[i].RT_mem_accesses.front();
        addr_set[record.addr]++;
      }
    }
  }

  void update_next_rt_accesses() {
    for (unsigned i=0; i<m_per_scalar_thread.size(); i++) {
      if (!m_per_scalar_thread[i].RT_mem_accesses.empty()) {
        RTMemoryTransactionRecord next_access = m_per_scalar_thread[i].RT_mem_accesses.front();
        
        // If "unmarked", this has not been added to queue yet (also make sure intersection is complete)
        if (next_access.status == RTMemStatus::RT_MEM_UNMARKED && m_per_scalar_thread[i].intersection_delay == 0) {
          std::pair<uint32_t, uint32_t> address_size_pair (next_access.addr, next_access.size);
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

  RTMemoryTransactionRecord get_next_rt_mem_transaction() {
    // Update the list of next accesses
    update_next_rt_accesses();
    RTMemoryTransactionRecord next_access;
    std::pair<uint32_t, uint32_t> address_size_pair;
    
    do {
      // Choose the next one on the list
      next_access = m_next_rt_accesses.front();
      m_next_rt_accesses.pop_front();
      
      address_size_pair = std::pair<uint32_t, uint32_t>(next_access.addr, next_access.size);
      
    // Check that the address hasn't already been sent
    } while (m_next_rt_accesses_set.find(address_size_pair) == m_next_rt_accesses_set.end());
    
    m_next_rt_accesses_set.erase(address_size_pair);
    
    //RT_DPRINTF("Next access chosen: 0x%x with %dB\n", next_access.address, next_access.size);
    return next_access;
  }

  // void undo_rt_access(uint32_t addr){
  //   std::pair<uint32_t, uint32_t> address_size_pair (addr, 32);
  //   assert (m_next_rt_accesses_set.find(address_size_pair) == m_next_rt_accesses_set.end());
    
  //   // Repackage address into a transaction record
  //   RTMemoryTransactionRecord mem_record;
  //   mem_record.addr = addr;
  //   mem_record.size = 32;
  //   // Already in queue
  //   mem_record.status = RT_MEM_AWAITING;

  //   m_next_rt_accesses.push_front(mem_record);
  //   m_next_rt_accesses_set.insert(address_size_pair);
  //   //RT_DPRINTF("UNDO: 0x%x added back to queue\n", addr);
  // }

  unsigned dec_thread_latency(std::deque<std::pair<uint32_t, uint32_t> > &store_queue) { 
    // Track number of threads performing intersection tests
    unsigned n_threads = 0;
    
    for (unsigned i=0; i<m_per_scalar_thread.size(); i++) {
      if (m_per_scalar_thread[i].intersection_delay > 0) {
        m_per_scalar_thread[i].intersection_delay--; 
        n_threads++;
        if (m_per_scalar_thread[i].intersection_delay == 0 && m_per_scalar_thread[i].ray_intersect) {
          // Temporary size
          // unsigned size = RT_WRITE_BACK_SIZE;

          // // Get an address to write to
          // void* next_buffer_addr = GPGPUSim_Context(GPGPU_Context())->get_device()->get_gpgpu()->gpu_malloc(size);
          // store_queue.push_back(std::pair<uint32_t, uint32_t>(m_uid, (uint32_t)next_buffer_addr));
          m_per_scalar_thread[i].ray_intersect = false;
          // RT_DPRINTF("Buffer store pushed for warp %d thread %d at 0x%x\n", m_uid, i, next_buffer_addr);

          // m_pending_writes.insert((uint32_t)next_buffer_addr);
        }
        
        // for(auto & store_transaction : m_per_scalar_thread[i].RT_store_transactions) {
        //   store_queue.push_back(std::pair<uint32_t, uint32_t>(m_uid, (new_addr_type)(store_transaction.address)));
        //   RT_DPRINTF("Buffer store pushed for warp %d thread %d at 0x%x\n", m_uid, i, store_transaction.address);

        //   assert(m_pending_writes.find((new_addr_type)store_transaction.address) == m_pending_writes.end());
        //   m_pending_writes.insert((new_addr_type)store_transaction.address);
        // }
        // m_per_scalar_thread[i].RT_store_transactions.clear();
      }
    }
    
    return n_threads;
  }
    
  unsigned process_returned_mem_access(uint32_t rsp_addr) {
    // Count how many threads used this mf
    uint32_t thread_found = 0;
    
    // Iterate through every thread in the warp
    for (unsigned i=0; i<m_per_scalar_thread.size(); i++) {
      if (process_returned_mem_access(i, rsp_addr)) {
        thread_found++;
      }
    }
    return thread_found;
  }

  bool process_returned_mem_access(unsigned tid, uint32_t rsp_addr) {
    bool thread_found = false;
    if (!m_per_scalar_thread[tid].RT_mem_accesses.empty()) {
      RTMemoryTransactionRecord &mem_record = m_per_scalar_thread[tid].RT_mem_accesses.front();
      
      if (mem_record.addr == rsp_addr) {
        // Set up delay of next intersection test
        unsigned n_delay_cycles = 4;
        m_per_scalar_thread[tid].intersection_delay += n_delay_cycles;
        
        // RT_DPRINTF("Thread %d collected all chunks for address 0x%x (size %d)\n", tid, mem_record.address, mem_record.size);
        // RT_DPRINTF("Processing data of transaction type %d for %d cycles.\n", mem_record.type, n_delay_cycles);
        m_per_scalar_thread[tid].RT_mem_accesses.pop_front();

        // Mark triangle hit to store to memory
        if (mem_record.type == TransactionType::BVH_QUAD_LEAF_HIT) {
          m_per_scalar_thread[tid].ray_intersect = true;
          //RT_DPRINTF("Buffer store detected for warp %d thread %d\n", m_uid, tid);
        }
        thread_found = true;
      }
      
      // If the RT_mem_accesses is now empty, then the last memory request has returned and the thread is almost done
      if (m_per_scalar_thread[tid].RT_mem_accesses.empty()) {
        // unsigned long long current_cycle =  GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle + GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
        // m_per_scalar_thread[tid].end_cycle = current_cycle + m_per_scalar_thread[tid].intersection_delay;
      }
    }
    return thread_found;
  }

  bool is_stalled() {
    // If there are still memory requests waiting to be processed, not stalled
    if (!m_next_rt_accesses_set.empty()) {
      return false;
    }
    // Otherwise check every thread
    for (unsigned i=0; i<m_per_scalar_thread.size(); i++) {
      if (!m_per_scalar_thread[i].RT_mem_accesses.empty()) {
        RTMemoryTransactionRecord mem_record = m_per_scalar_thread[i].RT_mem_accesses.front();
        
        // If there is an unprocessed record, not stalled
        if (mem_record.status == RT_MEM_UNMARKED && m_per_scalar_thread[i].intersection_delay == 0) return false;
      }
    }
    
    // Otherwise stalled
    // No queued addresses (m_next_rt_accesses_set), AND No new address is ready to be scheduled.
    return true;
  }

  bool rt_intersection_delay_done() { 
    bool done = true;
    for (unsigned i = 0; i < m_per_scalar_thread.size(); i++) {
      done &= (m_per_scalar_thread[i].intersection_delay == 0);
    }
    return done;
  }

  unsigned get_rt_active_threads() {
    assert(m_per_scalar_thread_valid);
    unsigned active_threads = 0;
    for (auto it=m_per_scalar_thread.begin(); it!=m_per_scalar_thread.end(); it++) {
      if (!it->RT_mem_accesses.empty()) {
        active_threads++;
      }
    }
    return active_threads;
  }

  std::deque<uint32_t> get_rt_active_thread_list() {
    assert(m_per_scalar_thread_valid);
    std::deque<uint32_t> active_threads;
    for (unsigned i=0; i<m_per_scalar_thread.size(); i++) {
      if (!m_per_scalar_thread[i].RT_mem_accesses.empty()) {
        active_threads.push_back(i);
      }
    }
    return active_threads;
  }

  void track_rt_cycles(bool active, ThreadMask &tmask) {
    bool stalled = is_stalled();
    unsigned warp_status = active ? warp_executing : stalled ? warp_stalled : warp_waiting;

    // Check progress of each thread
    for (unsigned i=0; i<m_per_scalar_thread.size(); i++) {
      // Only check active threads
      if (tmask.test(i)) {
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

  unsigned * get_latency_dist(unsigned i) {
    return (unsigned *)m_per_scalar_thread[i].status_num_cycles;
  }

  bool check_pending_writes(uint32_t addr) {
    if (m_pending_writes.find(addr) != m_pending_writes.end()) {
      m_pending_writes.erase(addr);
      return true;
    }
    else {
      return false;
    }
  }
};

};