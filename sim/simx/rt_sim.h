#include "instr_trace.h"

enum RT_STATUS {RT_MEM_UNMARKED, RT_MEM_AWAITING};

struct RTMemoryTransactionRecord {
    uint32_t address;
    uint32_t size;
    uint32_t type;
    RT_STATUS status;
};

struct RtuTraceData : public ITraceData {
    using Ptr = std::shared_ptr<RtuTraceData>;
    std::vector<std::vector<RTMemoryTransactionRecord>> mem_accesses;
    uint32_t pipeline_latency; 
    RtuTraceData(uint32_t num_threads = 0) : mem_addrs(num_threads) {}
};