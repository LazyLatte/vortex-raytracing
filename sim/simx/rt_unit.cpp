#include "rt_unit.h"
#include "rt_traversal.h"
#include "core.h"
#include "rt_sim.h"
#include <VX_config.h>
#include <vector>
#include <stack>
#include <unordered_map>
using namespace vortex;

enum ShaderType {MISS, CLOSET, INTERSECTION, ANY, NUM};

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
        , bvh_traverser_(simobject, dcrs)
        , shader_queues(ShaderType::NUM, ShaderQueue(32, NUM_RTU_LANES))
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

    void init_ray(std::vector<reg_data_t>& rd_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = ray_list_.allocate();
            rays_[rayID] = Ray();
            hits_[rayID] = Hit();
            traversal_trails_[rayID] = {};
            traversal_stacks_[rayID] = TraversalStack(RT_STACK_SIZE);
            rd_data[tid].u32 = rayID;
        } 
    }

    void set_ray_properties(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data, uint32_t axis){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t rs2 = rs2_data[tid].u32;
            uint32_t rs3 = rs3_data[tid].u32;
            
            float orig = *reinterpret_cast<float*>(&rs2);
            float dir = *reinterpret_cast<float*>(&rs3);

            switch(axis){
                case 0:
                    rays_[rayID].ro_x = orig;
                    rays_[rayID].rd_x = dir;
                    break;
                case 1:
                    rays_[rayID].ro_y = orig;
                    rays_[rayID].rd_y = dir;
                    break;
                case 2:
                    rays_[rayID].ro_z = orig;
                    rays_[rayID].rd_z = dir;
                    break;
                default: break;
            }

        }  
    }

    void traverse(uint32_t rayID, per_thread_info &thread_info){
        bool completed = bvh_traverser_.traverse(
            rays_[rayID], 
            hits_[rayID],
            traversal_trails_[rayID],
            traversal_stacks_[rayID],
            thread_info
        );
        
        if(completed){
            if(hits_[rayID].dist == LARGE_FLOAT){
                shader_queues[ShaderType::MISS].push(rayID);
            }else{
                shader_queues[ShaderType::CLOSET].push(rayID);
            }
        }else{
            shader_queues[ShaderType::ANY].push(rayID);
        }
    }

    void traverse(const std::vector<reg_data_t>& rs1_data, RtuTraceData* trace_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            traverse(rayID, trace_data->m_per_scalar_thread[tid]);
        }
    }

    ShaderType schedule_queue(){
        ShaderType targetType = ShaderType::MISS;
        if(shader_queues[ShaderType::CLOSET].size() > shader_queues[targetType].size()){
            targetType = ShaderType::CLOSET;
        }  

        if(shader_queues[ShaderType::ANY].size() > shader_queues[targetType].size()){
            targetType = ShaderType::ANY;
        }

        return targetType;
    }

    void get_work(std::vector<reg_data_t>& rd_data){
        if(shader_queues[ShaderType::MISS].empty() && 
            shader_queues[ShaderType::CLOSET].empty() &&
            shader_queues[ShaderType::ANY].empty()){
            for (uint32_t tid = 0; tid < num_lanes_; tid++) {
                rd_data[tid].u32 = 0;
            }
            return;
        }

        uint32_t type = schedule_queue();
        std::vector<uint32_t> targetQueue = shader_queues[type].pop();

        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            if(tid < targetQueue.size()){
                uint32_t rayID = targetQueue.at(tid);
                rd_data[tid].u32 = (1 << (28 + type)) | (rayID & 0x0FFFFFFF); 
            }else{
                rd_data[tid].u32 = (1 << (28 + type)); 
            }
        }
    }

    void get_attr(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, std::vector<reg_data_t>& rd_data){
        
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t attrID = rs2_data[tid].u32;

            switch(attrID){
                case VX_RT_RAY_RO_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&rays_[rayID].ro_x); break;
                case VX_RT_RAY_RO_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&rays_[rayID].ro_y); break;
                case VX_RT_RAY_RO_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&rays_[rayID].ro_z); break;
                case VX_RT_RAY_RD_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&rays_[rayID].rd_x); break;
                case VX_RT_RAY_RD_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&rays_[rayID].rd_y); break;
                case VX_RT_RAY_RD_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&rays_[rayID].rd_z); break;
                case VX_RT_RAY_BOUNCE: rd_data[tid].u32 = bounces_[rayID]; break;

                case VX_RT_HIT_DIST: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hits_[rayID].dist); break;
                case VX_RT_HIT_BX: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hits_[rayID].bx); break;
                case VX_RT_HIT_BY: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hits_[rayID].by); break;
                case VX_RT_HIT_BZ: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hits_[rayID].bz); break;
                case VX_RT_HIT_BLAS_IDX: rd_data[tid].u32 = hits_[rayID].blasIdx; break;
                case VX_RT_HIT_TRI_IDX: rd_data[tid].u32 = hits_[rayID].triIdx; break;

                case VX_RT_COLOR_R: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&colors_[rayID][0]); break;
                case VX_RT_COLOR_G: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&colors_[rayID][1]); break;
                case VX_RT_COLOR_B: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&colors_[rayID][2]); break;

                default: rd_data[tid].u32 = 0; break;
            }
        } 
    }

    void set_color(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, uint32_t ch){
        assert(ch == 0 || ch == 1 || ch ==2);
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t rs2 = rs2_data[tid].u32;

            float val = *reinterpret_cast<float*>(&rs2);
            colors_[rayID][ch] = val;
        }
    }

    void commit(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, RtuTraceData* trace_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t actionID = rs2_data[tid].u32;

            switch(actionID){
                case VX_RT_COMMIT_CONT: 
                    traverse(rayID, trace_data->m_per_scalar_thread[tid]);
                    break;
                case VX_RT_COMMIT_ACCEPT: 
                    hits_[rayID].dist = hits_[rayID].pending_dist;
                    traverse(rayID, trace_data->m_per_scalar_thread[tid]);
                    break;
                case VX_RT_COMMIT_TERM: 
                    ray_list_.free(rayID);
                    break;
                default: break;
            }
        }
    }

    void set_ray_bounce(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t bounce = rs2_data[tid].u32;
            bounces_[rayID] = bounce;
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

    BVHTraverser bvh_traverser_;

    RayList ray_list_;
    std::unordered_map<uint32_t, Ray> rays_;
    std::unordered_map<uint32_t, Hit> hits_;
    std::unordered_map<uint32_t, TraversalTrail> traversal_trails_;
    std::unordered_map<uint32_t, TraversalStack> traversal_stacks_;
    
    std::unordered_map<uint32_t, std::array<float, 3>> colors_;

    std::unordered_map<uint32_t, uint32_t> bounces_;

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

void RTUnit::init_ray(std::vector<reg_data_t>& rd_data){
    impl_->init_ray(rd_data);
}

void RTUnit::set_ray_properties(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data, uint32_t axis){
    impl_->set_ray_properties(rs1_data, rs2_data, rs3_data, axis);
}

void RTUnit::set_ray_bounce(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data){
    impl_->set_ray_bounce(rs1_data, rs2_data);
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

void RTUnit::set_color(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, uint32_t ch){
    impl_->set_color(rs1_data, rs2_data, ch);
}

void RTUnit::commit(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, RtuTraceData* trace_data){
    impl_->commit(rs1_data, rs2_data, trace_data);
}