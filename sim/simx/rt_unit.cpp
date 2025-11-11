#include "rt_unit.h"
#include "rt_traversal.h"
#include "core.h"
#include <VX_config.h>
#include <vector>

#define MEM_QUEUE_SIZE 32
using namespace vortex;

class RTUnit::Impl {
public:
    Impl(RTUnit* simobject, const Arch &arch, const DCRS &dcrs, Core* core)
        : simobject_(simobject)
        , core_(core)
        , arch_(arch)
        , dcrs_(dcrs)
        , perf_stats_()
        , num_blocks_(NUM_RTU_BLOCKS)
        , num_lanes_(NUM_RTU_LANES)
        , ray_buffers_(arch.num_warps())
        , pending_reqs_(MEM_QUEUE_SIZE)
    {
        for (uint32_t i = 0; i < arch.num_warps(); ++i) {
            ray_buffers_.at(i).resize(arch.num_threads());
        }
    }

    ~Impl() {}

    void reset() {
        pending_reqs_.clear();
        perf_stats_ = PerfStats();
    }

    void tick() {
        // response
        for(uint32_t iw = 0; iw < num_blocks_; iw++){
            for (uint32_t t = 0; t < num_lanes_; t++) {
                auto& dcache_rsp_port = simobject_->MemRsps.at(iw).at(t);
                if (dcache_rsp_port.empty())
                    continue;
                auto& mem_rsp = dcache_rsp_port.front();
                auto& entry = pending_reqs_.at(mem_rsp.tag);
                auto trace = entry.trace;
                DT(3, simobject_->name() << "-rt-rsp: tag=" << mem_rsp.tag << ", tid=" << t << ", " << *trace);
                assert(entry.count);
                --entry.count; // track remaining addresses
                if (0 == entry.count) {
                    simobject_->Outputs.at(iw).push(trace, 1);
                    pending_reqs_.release(mem_rsp.tag);
                }
                dcache_rsp_port.pop();
            }
        }


        for (int i = 0, n = pending_reqs_.size(); i < n; ++i) {
            if (pending_reqs_.contains(i))
                perf_stats_.latency += pending_reqs_.at(i).count;
        }

        // request
        for (uint32_t iw = 0; iw  <num_blocks_; ++iw){
            auto& input = simobject_->Inputs.at(iw);

            if (input.empty())
                continue;

            auto trace = input.front();

            if (pending_reqs_.full()) {
                if (!trace->log_once(true)) {
                    DT(3, "*** " << simobject_->name() << "-rt-queue-stall: " << *trace);
                }
                ++perf_stats_.stalls;
                return;
            } else {
                trace->log_once(false);
            }

            auto trace_data = std::dynamic_pointer_cast<RtuTraceData>(trace->data);

            uint32_t addr_count = 0;
            for (auto& mem_addr : trace_data->mem_addrs) {
                addr_count += mem_addr.size;
            }

            if (addr_count != 0) {
                auto tag = pending_reqs_.allocate({trace, addr_count});
                for (uint32_t t = 0; t < num_lanes_; ++t) {
                    if (!trace->tmask.test(t))
                        continue;

                    auto& dcache_req_port = simobject_->MemReqs.at(iw).at(t);
                    for (auto& mem_addr : trace_data->mem_addrs) {
                        MemReq mem_req;
                        mem_req.addr  = mem_addr.addr;
                        mem_req.write = false;
                        mem_req.tag   = tag;
                        mem_req.cid   = trace->cid;
                        mem_req.uuid  = trace->uuid;
                        dcache_req_port.push(mem_req, 4);
                        DT(3, simobject_->name() << "-rt-req: addr=0x" << std::hex << mem_addr.addr << ", tag=" << tag
                            << ", tid=" << t << ", "<< trace);
                        ++perf_stats_.reads;
                    }
                }
            } else {
                simobject_->Outputs.at(iw).push(trace, 1);
            }

            input.pop();

            //simobject_->Outputs.at(iw).push(trace, trace_data->pipeline_latency);
        }
    }

    const PerfStats& perf_stats() const {
        return perf_stats_;
    }

    void get_csr(uint32_t addr, uint32_t wid, uint32_t tid, Word* value){
        switch(addr){
            case VX_CSR_RTX_RO1:
                *value = *reinterpret_cast<Word*>(&ray_buffers_.at(wid).at(tid).ray.ro_x);
                break;
            case VX_CSR_RTX_RO2:
                *value = *reinterpret_cast<Word*>(&ray_buffers_.at(wid).at(tid).ray.ro_y);
                break;
            case VX_CSR_RTX_RO3:
                *value = *reinterpret_cast<Word*>(&ray_buffers_.at(wid).at(tid).ray.ro_z);
                break;
            case VX_CSR_RTX_RD1:
                *value = *reinterpret_cast<Word*>(&ray_buffers_.at(wid).at(tid).ray.rd_x);
                break;
            case VX_CSR_RTX_RD2:
                *value = *reinterpret_cast<Word*>(&ray_buffers_.at(wid).at(tid).ray.rd_y);
                break;
            case VX_CSR_RTX_RD3:
                *value = *reinterpret_cast<Word*>(&ray_buffers_.at(wid).at(tid).ray.rd_z);
                break;
            case VX_CSR_RTX_BCOORDS1:
                *value = *reinterpret_cast<Word*>(&ray_buffers_.at(wid).at(tid).hit.bx);
                break;
            case VX_CSR_RTX_BCOORDS2:
                *value = *reinterpret_cast<Word*>(&ray_buffers_.at(wid).at(tid).hit.by);
                break;
            case VX_CSR_RTX_BCOORDS3:
                *value = *reinterpret_cast<Word*>(&ray_buffers_.at(wid).at(tid).hit.bz);
                break;
            case VX_CSR_RTX_BLAS_IDX:
                *value = ray_buffers_.at(wid).at(tid).hit.blasIdx;
                break;
            case VX_CSR_RTX_TRI_IDX:
                *value = ray_buffers_.at(wid).at(tid).hit.triIdx;
                break;
            default: std::abort();
        }
    }

    void set_csr(uint32_t addr, uint32_t wid, uint32_t tid, Word value){
        switch(addr){
            case VX_CSR_RTX_RO1:
                ray_buffers_.at(wid).at(tid).ray.ro_x = *reinterpret_cast<float*>(&value);
                break;
            case VX_CSR_RTX_RO2:
                ray_buffers_.at(wid).at(tid).ray.ro_y = *reinterpret_cast<float*>(&value);
                break;
            case VX_CSR_RTX_RO3:
                ray_buffers_.at(wid).at(tid).ray.ro_z = *reinterpret_cast<float*>(&value);
                break;
            case VX_CSR_RTX_RD1:
                ray_buffers_.at(wid).at(tid).ray.rd_x = *reinterpret_cast<float*>(&value);
                break;
            case VX_CSR_RTX_RD2:
                ray_buffers_.at(wid).at(tid).ray.rd_y = *reinterpret_cast<float*>(&value);
                break;
            case VX_CSR_RTX_RD3:
                ray_buffers_.at(wid).at(tid).ray.rd_z = *reinterpret_cast<float*>(&value);
                break;
            case VX_CSR_RTX_BCOORDS1:
                ray_buffers_.at(wid).at(tid).hit.bx = *reinterpret_cast<float*>(&value);
                break;
            case VX_CSR_RTX_BCOORDS2:
                ray_buffers_.at(wid).at(tid).hit.by = *reinterpret_cast<float*>(&value);
                break;
            case VX_CSR_RTX_BCOORDS3:
                ray_buffers_.at(wid).at(tid).hit.bz = *reinterpret_cast<float*>(&value);
                break;
            case VX_CSR_RTX_BLAS_IDX:
                ray_buffers_.at(wid).at(tid).hit.blasIdx = value;
                break;
            case VX_CSR_RTX_TRI_IDX:
                ray_buffers_.at(wid).at(tid).hit.triIdx = value;
                break;
            default: std::abort();
        }
        
    } 

    void dcache_read(void* data, uint64_t addr, uint32_t size) {
        core_->dcache_read(data, addr, size);
    }

    void dcache_write(const void* data, uint64_t addr, uint32_t size) {
        core_->dcache_write(data, addr, size);
    }

    void traverse(uint32_t wid, std::vector<reg_data_t>& rd_data, RtuTraceData* trace_data){
        uint32_t tlas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TLAS_PTR);
        uint32_t blas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BLAS_PTR);
        //uint32_t bvh_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BVH_PTR);
        uint32_t qBvh_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_QBVH_PTR);
        uint32_t tri_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_PTR);
        uint32_t tri_idx_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_IDX_PTR);

        RestartTrailTraversal rtt(tlas_ptr, blas_ptr, qBvh_ptr, tri_ptr, tri_idx_ptr, simobject_);

        trace_data->pipeline_latency = 0;

        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            RayBuffer &ray_buffer = ray_buffers_.at(wid).at(tid);
            ray_buffer.hit.dist = LARGE_FLOAT;
            rtt.traverse(ray_buffer, trace_data);
            rd_data[tid].i = *reinterpret_cast<uint32_t*>(&ray_buffer.hit.dist);
        }
    }

private:
    RTUnit*       simobject_;
    Core*         core_;
    const Arch&   arch_;
    const DCRS&   dcrs_;
    PerfStats     perf_stats_;

    uint32_t num_blocks_;
    uint32_t num_lanes_;
    std::vector<std::vector<RayBuffer>> ray_buffers_;

    struct pending_req_t {
        instr_trace_t* trace;
        uint32_t count;
    };

    HashTable<pending_req_t> pending_reqs_;
};

RTUnit::RTUnit(const SimContext &ctx, const char* name, const Arch &arch, const DCRS &dcrs, Core* core)
    : SimObject<RTUnit>(ctx, name)
    , Inputs(ISSUE_WIDTH, this)
	, Outputs(ISSUE_WIDTH, this)
    , MemReqs(NUM_RTU_BLOCKS, std::vector<SimPort<MemReq>>(NUM_RTU_LANES, this))
    , MemRsps(NUM_RTU_BLOCKS, std::vector<SimPort<MemRsp>>(NUM_RTU_LANES, this))
	, impl_(new Impl(this, arch, dcrs, core))
{}

RTUnit::~RTUnit() {
  delete impl_;
}

void RTUnit::reset() {
  impl_->reset();
}

void RTUnit::tick() {
  impl_->tick();
}

const RTUnit::PerfStats &RTUnit::perf_stats() const {
	return impl_->perf_stats();
}

void RTUnit::get_csr(uint32_t addr, uint32_t wid, uint32_t tid, Word* value){
    impl_->get_csr(addr, wid, tid, value);    
} 

void RTUnit::set_csr(uint32_t addr, uint32_t wid, uint32_t tid, Word value){
    impl_->set_csr(addr, wid, tid, value);    
} 

void RTUnit::dcache_read(void* data, uint64_t addr, uint32_t size){
    impl_->dcache_read(data, addr, size);
}

void RTUnit::dcache_write(const void* data, uint64_t addr, uint32_t size){
    impl_->dcache_write(data, addr, size);
}

void RTUnit::traverse(uint32_t wid, std::vector<reg_data_t>& rd_data, RtuTraceData* trace_data){
    impl_->traverse(wid, rd_data, trace_data);
}