#include "rt_unit.h"
#include "rt_traversal.h"
#include "core.h"
#include "rt_sim.h"
#include <VX_config.h>
#include <vector>
#include <stack>
#include <unordered_map>
using namespace vortex;

enum ShaderType {MISS, CLOSET, INTERSECTION, ANY};
enum AttrID {RO_X, RO_Y, RO_Z, RD_X, RD_Y, RD_Z, DIST, BX, BY, BZ, BLAS_IDX, TRI_IDX};

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
        , shader_queues(4, ShaderQueue(32, NUM_RTU_LANES)) // miss, closet-hit, intersection, any-hit
    {}

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

    void dcache_read(void* data, uint64_t addr, uint32_t size) {
        core_->dcache_read(data, addr, size);
    }

    void dcache_write(const void* data, uint64_t addr, uint32_t size) {
        core_->dcache_write(data, addr, size);
    }

    void create_ray(std::vector<reg_data_t>& rd_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            rd_data[tid].u32 = ray_buffers_.allocate();
        } 
    }

    void set_ray_x(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t rs2 = rs2_data[tid].u32;
            uint32_t rs3 = rs3_data[tid].u32;
            
            float ro_x = *reinterpret_cast<float*>(&rs2);
            float rd_x = *reinterpret_cast<float*>(&rs3);

            ray_buffers_.set_ray_x(rayID, ro_x, rd_x);
        }  
    }

    void set_ray_y(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t rs2 = rs2_data[tid].u32;
            uint32_t rs3 = rs3_data[tid].u32;
            
            float ro_y = *reinterpret_cast<float*>(&rs2);
            float rd_y = *reinterpret_cast<float*>(&rs3);

            ray_buffers_.set_ray_y(rayID, ro_y, rd_y);
        }  
    }

    void set_ray_z(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t rs2 = rs2_data[tid].u32;
            uint32_t rs3 = rs3_data[tid].u32;
            
            float ro_z = *reinterpret_cast<float*>(&rs2);
            float rd_z = *reinterpret_cast<float*>(&rs3);

            ray_buffers_.set_ray_z(rayID, ro_z, rd_z);
        }  
    }

    void traverse(const std::vector<reg_data_t>& rs1_data, RtuTraceData* trace_data){
        uint32_t tlas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TLAS_PTR);
        uint32_t blas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BLAS_PTR);
        uint32_t qBvh_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_QBVH_PTR);
        uint32_t tri_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_PTR);
        uint32_t tri_idx_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_IDX_PTR);

        RestartTrailTraversal rtt(tlas_ptr, blas_ptr, qBvh_ptr, tri_ptr, tri_idx_ptr, simobject_);

        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            std::pair<Ray, Hit>& raybuf = ray_buffers_.get(rayID);
            // RayBuffer &ray_buffer = ray_buffers_.at(wid).at(tid);
            // ray_buffer.hit.dist = LARGE_FLOAT;
            rtt.traverse(raybuf, trace_data->m_per_scalar_thread[tid]);

            //distances_.at(wid).at(tid) = *reinterpret_cast<uint32_t*>(&ray_buffer.hit.dist);
            
            if(raybuf.second.dist == LARGE_FLOAT){
                shader_queues[ShaderType::MISS].push(rayID);
            }else{
                shader_queues[ShaderType::CLOSET].push(rayID);
            }
        }
    }

    void get_work(std::vector<reg_data_t>& rd_data){
        if(shader_queues[ShaderType::MISS].empty()){
            for (uint32_t tid = 0; tid < num_lanes_; tid++) {
                rd_data[tid].u32 = 0xFFFFFFFF; // temp value!!! 
            }
            return;
        }

        std::vector<uint32_t> target = shader_queues[ShaderType::MISS].pop();

        uint32_t type = ShaderType::MISS;

        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            if(tid < target.size()){
                uint32_t rayID = target.at(tid);
                rd_data[tid].u32 = (type << 30) | (rayID & 0x3FFFFFFF); 
            }else{
                rd_data[tid].u32 = (type << 30) | 0x3FFFFFFF; 
            }
        }

        
    }

    void get_attr(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, std::vector<reg_data_t>& rd_data){
        
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t attrID = rs2_data[tid].u32;

            std::pair<Ray, Hit>& raybuf = ray_buffers_.get(rayID);
            switch(attrID){
                case AttrID::RO_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&raybuf.first.ro_x); break;
                case AttrID::RO_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&raybuf.first.ro_y); break;
                case AttrID::RO_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&raybuf.first.ro_z); break;
                case AttrID::RD_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&raybuf.first.rd_x); break;
                case AttrID::RD_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&raybuf.first.rd_y); break;
                case AttrID::RD_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&raybuf.first.rd_z); break;

                case AttrID::DIST: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&raybuf.second.dist); break;
                case AttrID::BX: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&raybuf.second.bx); break;
                case AttrID::BY: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&raybuf.second.by); break;
                case AttrID::BZ: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&raybuf.second.bz); break;
                case AttrID::BLAS_IDX: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&raybuf.second.blasIdx); break;
                case AttrID::TRI_IDX: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&raybuf.second.triIdx); break;
                default: rd_data[tid].u32 = 0; break;
            }
        } 
    }

    void set_color(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t rs2 = rs2_data[tid].u32;
            uint32_t rs3 = rs3_data[tid].u32;

            float r = *reinterpret_cast<float*>(&rs2);
            float g = *reinterpret_cast<float*>(&rs3);
            float b = 0.0f;
            colors_[rayID] = {r, g, b};
        }
    }

    void get_color(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, std::vector<reg_data_t>& rd_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t idx = rs2_data[tid].u32;
            float value = colors_[rayID][idx];
            rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&value);
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

    RayBuffer ray_buffers_;
    std::unordered_map<uint32_t, std::array<float, 3>> colors_;
    std::vector<ShaderQueue> shader_queues;
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

void RTUnit::dcache_read(void* data, uint64_t addr, uint32_t size){
    impl_->dcache_read(data, addr, size);
}

void RTUnit::dcache_write(const void* data, uint64_t addr, uint32_t size){
    impl_->dcache_write(data, addr, size);
}

void RTUnit::create_ray(std::vector<reg_data_t>& rd_data){
    impl_->create_ray(rd_data);
}

void RTUnit::set_ray_x(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data){
    impl_->set_ray_x(rs1_data, rs2_data, rs3_data);
}

void RTUnit::set_ray_y(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data){
    impl_->set_ray_y(rs1_data, rs2_data, rs3_data);
}

void RTUnit::set_ray_z(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data){
    impl_->set_ray_z(rs1_data, rs2_data, rs3_data);
}

void RTUnit::traverse(const std::vector<reg_data_t>& rs1_data, RtuTraceData* trace_data){
    impl_->traverse(rs1_data, trace_data);
}

void RTUnit::get_work(std::vector<reg_data_t>& rd_data){
    impl_->get_work(rd_data);
}

void RTUnit::get_attr(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, std::vector<reg_data_t>& rd_data){
    impl_->get_attr(rs1_data, rs2_data, rd_data);
}

void RTUnit::set_color(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data){
    impl_->set_color(rs1_data, rs2_data, rs3_data);
}

void RTUnit::get_color(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, std::vector<reg_data_t>& rd_data){
    impl_->get_color(rs1_data, rs2_data, rd_data);
}