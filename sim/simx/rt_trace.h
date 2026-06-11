#pragma once
#include "instr_trace.h"
#include <set>
#include <unordered_map>

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
  waiting_shader,
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
  INTERSECTION_TABLE_LOAD,
  UNDEFINED,
};

enum class StoreTransactionType {TRAVERSAL_RESULTS, INTERSECTION_TABLE_STORE};

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
  uint32_t addr;
  uint32_t size;
  StoreTransactionType type;
  MemoryStoreTransactionRecord(){}
  MemoryStoreTransactionRecord(uint32_t addr, uint32_t size, StoreTransactionType type)
  : addr(addr), size(size), type(type) {}
};

struct per_thread_info {      
  per_thread_info(){}
  ~per_thread_info(){}

  std::deque<RTMemoryTransactionRecord> RT_mem_accesses;
  std::vector<MemoryStoreTransactionRecord> RT_store_transactions;
  bool terminate = false;
  uint32_t rayID = 0xFFFFFFFF;
  bool ray_intersect = false;
  unsigned intersection_delay = 0;
  unsigned status_num_cycles[warp_statuses][ray_statuses] = {};
};

//warp_instr_t
struct RtuTraceData : public ITraceData {
  using Ptr = std::shared_ptr<RtuTraceData>;
  std::vector<per_thread_info> m_per_scalar_thread;

  std::deque<RTMemoryTransactionRecord> m_next_rt_accesses;
  std::set<std::pair<uint32_t, uint32_t>> m_next_rt_accesses_set;
  std::set<uint32_t> m_pending_writes;
  RtuTraceData(uint32_t num_threads = 0) : m_per_scalar_thread(num_threads){}

  bool rt_traversal_terminate(){
    for (unsigned i = 0; i < m_per_scalar_thread.size(); i++) {
      if(!m_per_scalar_thread[i].terminate){
        return false;
      }
    }
    return true;
  }

  bool has_pending_writes() { return !m_pending_writes.empty(); }
  bool rt_mem_accesses_empty(){
    bool empty = true;
    for (unsigned i = 0; i < m_per_scalar_thread.size(); i++) {
      empty &= m_per_scalar_thread[i].RT_mem_accesses.empty();
    }
    empty &= m_next_rt_accesses_set.empty();
    return empty;
  }

  void num_unique_mem_access(std::unordered_map<uint32_t, uint32_t> &addr_set){
    for (unsigned i = 0; i < m_per_scalar_thread.size(); i++) {
      if (!m_per_scalar_thread[i].RT_mem_accesses.empty()) {
        RTMemoryTransactionRecord record = m_per_scalar_thread[i].RT_mem_accesses.front();
        addr_set[record.addr]++;
      }
    }
  }

  unsigned* get_latency_dist(unsigned i) { return (unsigned *)m_per_scalar_thread[i].status_num_cycles; }

  void update_next_rt_accesses();
  RTMemoryTransactionRecord get_next_rt_mem_transaction();
  unsigned process_returned_mem_access(uint32_t rsp_addr);
  bool process_returned_mem_access(unsigned tid, uint32_t rsp_addr);
  unsigned dec_thread_latency(instr_trace_t* trace, std::deque<std::pair<instr_trace_t*, MemoryStoreTransactionRecord>> &store_queue);


  bool is_stalled();

  bool rt_intersection_delay_done();
  unsigned get_rt_active_threads();

  std::deque<uint32_t> get_rt_active_thread_list();

  void track_rt_cycles(bool active, ThreadMask &tmask);
  bool check_pending_writes(uint32_t addr);
};

};