#include "instr_trace.h"

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

struct RTMemoryTransactionRecord {
    uint32_t addr;
    uint32_t size;
    uint32_t type;
    RTMemStatus status;
};

struct per_thread_info {
  per_thread_info();
  ~per_thread_info();
  dram_callback_t callback;
  new_addr_type
      memreqaddr[MAX_ACCESSES_PER_INSN_PER_THREAD];  // effective address,
                                                      // upto 8 different
                                                      // requests (to support
                                                      // 32B access in 8 chunks
                                                      // of 4B each)
                                                  
  // RT variables    
  std::deque<RTMemoryTransactionRecord> RT_mem_accesses;
  //std::vector<MemoryStoreTransactionRecord> RT_store_transactions;
  bool ray_intersect = false;
  Ray ray_properties;
  unsigned intersection_delay;
  unsigned long long end_cycle;
  unsigned status_num_cycles[warp_statuses][ray_statuses] = {};
  unsigned m_uid;
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

  RtuTraceData(uint32_t num_threads = 0) : m_per_scalar_thread(num_threads) {}

  bool rt_mem_accesses_empty(){
    for (unsigned i=0; i<num_lanes_; i++) {
      if(!m_per_scalar_thread[i].RT_mem_accesses.empty()){
        return false;
      }
    }
    return true;
  }

  void update_next_rt_accesses() {
    for (unsigned i=0; i<num_lanes_; i++) {
      if (!m_per_scalar_thread[i].RT_mem_accesses.empty()) {
        RTMemoryTransactionRecord next_access = m_per_scalar_thread[i].RT_mem_accesses.front();
        
        // If "unmarked", this has not been added to queue yet (also make sure intersection is complete)
        if (next_access.status == RTMemStatus::RT_MEM_UNMARKED && m_per_scalar_thread[i].intersection_delay == 0) {
          std::pair<uint32_t, uint32_t> address_size_pair (next_access.address, next_access.size);
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
      
      address_size_pair = std::pair<uint32_t, uint32_t>(next_access.address, next_access.size);
      
    // Check that the address hasn't already been sent
    } while (m_next_rt_accesses_set.find(address_size_pair) == m_next_rt_accesses_set.end());
    
    m_next_rt_accesses_set.erase(address_size_pair);
    
    //RT_DPRINTF("Next access chosen: 0x%x with %dB\n", next_access.address, next_access.size);
    return next_access;
  }

  unsigned dec_thread_latency(std::deque<std::pair<uint32_t, uint32_t> > &store_queue) { 
    // Track number of threads performing intersection tests
    unsigned n_threads = 0;
    
    for (unsigned i=0; i<num_lanes_; i++) {
      if (m_per_scalar_thread[i].intersection_delay > 0) {
        m_per_scalar_thread[i].intersection_delay--; 
        n_threads++;
        if (m_per_scalar_thread[i].intersection_delay == 0 && m_per_scalar_thread[i].ray_intersect) {
          // Temporary size
          // unsigned size = RT_WRITE_BACK_SIZE;

          // // Get an address to write to
          // void* next_buffer_addr = GPGPUSim_Context(GPGPU_Context())->get_device()->get_gpgpu()->gpu_malloc(size);
          // store_queue.push_back(std::pair<unsigned, new_addr_type>(m_uid, (new_addr_type)next_buffer_addr));
          m_per_scalar_thread[i].ray_intersect = false;
          RT_DPRINTF("Buffer store pushed for warp %d thread %d at 0x%x\n", m_uid, i, next_buffer_addr);

          //m_pending_writes.insert((new_addr_type)next_buffer_addr);
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
    

};
