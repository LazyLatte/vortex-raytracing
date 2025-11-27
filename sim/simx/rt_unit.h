#pragma once
#include <simobject.h>
#include "arch.h"
#include "dcrs.h"
#include "instr_trace.h"
#include <vector>

namespace vortex {

class Core;

class RTUnit : public SimObject<RTUnit> {
public:
	struct PerfStats {
    uint64_t stalls;
    uint64_t reads;
    uint64_t latency;

    PerfStats()
        : stalls(0)
        , reads(0)
        , latency(0)
    {}

    PerfStats& operator+=(const PerfStats& rhs) {
        this->reads   += rhs.reads;
        this->latency += rhs.latency;
        this->stalls  += rhs.stalls;
        return *this;
    }
	};
  
  std::vector<SimPort<instr_trace_t*>> Inputs; 
  std::vector<SimPort<instr_trace_t*>> Outputs; 
  RTUnit(const SimContext &ctx, const char* name, const Arch &arch, const DCRS &dcrs, Core* core);
  ~RTUnit();
  void reset();
  void tick();

  void get_csr(uint32_t addr, uint32_t wid, uint32_t tid, Word* value);
  void set_csr(uint32_t addr, uint32_t wid, uint32_t tid, Word value);

  void dcache_read(void* data, uint64_t addr, uint32_t size);
  void dcache_write(const void* data, uint64_t addr, uint32_t size);

  void traverse(uint32_t wid, std::vector<reg_data_t>& rd_data, RtuTraceData* trace_data);
  const PerfStats& perf_stats() const;
private:
	class Impl;
	Impl* impl_;
};

}