#include "rt_trace.h"

using namespace vortex;

std::unordered_map<TransactionType, unsigned> transaction_latencies = {
    {TransactionType::BVH_STRUCTURE, 1},
    {TransactionType::BVH_INTERNAL_NODE, RT_BVH_INTERNAL_NODE_LATENCY},
    {TransactionType::BVH_INSTANCE_LEAF, RT_BVH_INSTANCE_LEAF_LATENCY},
    {TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR, 1},
    {TransactionType::BVH_QUAD_LEAF, RT_BVH_QUAD_LEAF_LATENCY},
    {TransactionType::BVH_QUAD_LEAF_HIT, RT_BVH_QUAD_LEAF_HIT_LATENCY},
    {TransactionType::BVH_PROCEDURAL_LEAF, 1},
    {TransactionType::INTERSECTION_TABLE_LOAD, 1}
};

void RtuTraceData::update_next_rt_accesses() {
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

RTMemoryTransactionRecord RtuTraceData::get_next_rt_mem_transaction() {
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

    return next_access;
}

unsigned RtuTraceData::dec_thread_latency(instr_trace_t* trace, std::deque<std::pair<instr_trace_t*, MemoryStoreTransactionRecord>> &store_queue) { 
    // Track number of threads performing intersection tests
    unsigned n_threads = 0;

    for (unsigned i=0; i<m_per_scalar_thread.size(); i++) {
        if (m_per_scalar_thread[i].intersection_delay > 0) {
            m_per_scalar_thread[i].intersection_delay--; 
            n_threads++;
            if(m_per_scalar_thread[i].intersection_delay == 0 && m_per_scalar_thread[i].RT_mem_accesses.empty()){
                for(auto & store_transaction : m_per_scalar_thread[i].RT_store_transactions) {
                    store_queue.push_back(
                        std::pair<instr_trace_t*, MemoryStoreTransactionRecord>(trace, store_transaction)
                    );

                    assert(m_pending_writes.find(store_transaction.addr) == m_pending_writes.end());
                    m_pending_writes.insert(store_transaction.addr);
                }
            }

            // if (m_per_scalar_thread[i].intersection_delay == 0 && m_per_scalar_thread[i].ray_intersect) {
            //     m_per_scalar_thread[i].ray_intersect = false;
            //     // Buffer store??
            // }
        }
    }

    return n_threads;
}
    
unsigned RtuTraceData::process_returned_mem_access(uint32_t rsp_addr) {
    // Count how many threads used this addr
    uint32_t thread_found = 0;
    for (unsigned i=0; i<m_per_scalar_thread.size(); i++) {
        if (process_returned_mem_access(i, rsp_addr)) {
            thread_found++;
        }
    }
    return thread_found;
}

bool RtuTraceData::process_returned_mem_access(unsigned tid, uint32_t rsp_addr) {
    bool thread_found = false;
    if (!m_per_scalar_thread[tid].RT_mem_accesses.empty()) {
        RTMemoryTransactionRecord &mem_record = m_per_scalar_thread[tid].RT_mem_accesses.front();
        
        if (mem_record.addr == rsp_addr) {
            unsigned n_delay_cycles = transaction_latencies[mem_record.type];
            m_per_scalar_thread[tid].intersection_delay += n_delay_cycles;
            m_per_scalar_thread[tid].RT_mem_accesses.pop_front();

            // Mark triangle hit to store to memory
            // if (mem_record.type == TransactionType::BVH_QUAD_LEAF_HIT) {
            //     m_per_scalar_thread[tid].ray_intersect = true;
            // }

            thread_found = true;
        }
    }
    return thread_found;
}

bool RtuTraceData::is_stalled() {
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

bool RtuTraceData::rt_intersection_delay_done() { 
    bool done = true;
    for (unsigned i = 0; i < m_per_scalar_thread.size(); i++) {
        done &= (m_per_scalar_thread[i].intersection_delay == 0);
    }
    return done;
}

unsigned RtuTraceData::get_rt_active_threads() {
    //assert(m_per_scalar_thread_valid);
    unsigned active_threads = 0;
    for (auto it=m_per_scalar_thread.begin(); it!=m_per_scalar_thread.end(); it++) {
        if (!it->RT_mem_accesses.empty()) {
        active_threads++;
        }
    }
    return active_threads;
}

std::deque<uint32_t> RtuTraceData::get_rt_active_thread_list() {
    //assert(m_per_scalar_thread_valid);
    std::deque<uint32_t> active_threads;
    for (unsigned i=0; i<m_per_scalar_thread.size(); i++) {
        if (!m_per_scalar_thread[i].RT_mem_accesses.empty()) {
        active_threads.push_back(i);
        }
    }
    return active_threads;
}

void RtuTraceData::track_rt_cycles(bool active, ThreadMask &tmask) {
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

bool RtuTraceData::check_pending_writes(uint32_t addr) {
    if (m_pending_writes.find(addr) != m_pending_writes.end()) {
        m_pending_writes.erase(addr);
        return true;
    }else{
        return false;
    }
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