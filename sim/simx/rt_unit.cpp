#include "rt_unit.h"
#include "rt_traversal.h"
#include "core.h"
#include "rt_sim.h"
#include <VX_config.h>
#include <vector>

using namespace vortex;

class RTUnit::Impl {
public:
    Impl(RTUnit* simobject, const Arch &arch, const DCRS &dcrs, Core* core)
        : simobject_(simobject)
        , rt_sim_(new RTSim(simobject))
        , core_(core)
        , arch_(arch)
        , dcrs_(dcrs)
        , num_blocks_(NUM_RTU_BLOCKS)
        , num_lanes_(NUM_RTU_LANES)
        , ray_buffers_(arch.num_warps())
    {
        for (uint32_t i = 0; i < arch.num_warps(); ++i) {
            ray_buffers_.at(i).resize(arch.num_threads());
        }
    }

    ~Impl() {
        delete rt_sim_;
    }

    void reset() {
        rt_sim_->reset();
    }

    void tick() {
        rt_sim_->tick();
    }

    const PerfStats& perf_stats() const {
        return rt_sim_->perf_stats();
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

        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            RayBuffer &ray_buffer = ray_buffers_.at(wid).at(tid);
            ray_buffer.hit.dist = LARGE_FLOAT;
            rtt.traverse(ray_buffer, trace_data->m_per_scalar_thread[tid]);
            rd_data[tid].i = *reinterpret_cast<uint32_t*>(&ray_buffer.hit.dist);
        }
    }

private:
    RTUnit*       simobject_;
    RTSim*        rt_sim_;
    Core*         core_;
    const Arch&   arch_;
    const DCRS&   dcrs_;

    uint32_t num_blocks_;
    uint32_t num_lanes_;
    std::vector<std::vector<RayBuffer>> ray_buffers_;
};

RTUnit::RTUnit(const SimContext &ctx, const char* name, const Arch &arch, const DCRS &dcrs, Core* core)
    : SimObject<RTUnit>(ctx, name)
    , Inputs(ISSUE_WIDTH, this)
	, Outputs(ISSUE_WIDTH, this)
	, impl_(new Impl(this, arch, dcrs, core))
    , rtu_dcache_req_out(NUM_RTU_BLOCKS, std::vector<SimPort<MemReq>>(NUM_RTU_LANES, this))
    , rtu_dcache_rsp_in(NUM_RTU_BLOCKS, std::vector<SimPort<MemRsp>>(NUM_RTU_LANES, this))
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