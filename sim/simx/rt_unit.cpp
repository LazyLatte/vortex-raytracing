#include "rt_unit.h"
#include "rt_traversal.h"
#include "rt_sim.h"
#include "core.h"

#define SHADER_QUEUE_CAPACITY 1024

using namespace vortex;

enum ShaderType {MISS, CLOSET, INTERSECTION, ANY, ShaderTypes};

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
        , cur_rayid_(1)
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
            uint32_t rayID = cur_rayid_++;
            if(rayID == 0xF0000000) rayID = 1;
            rays_[rayID] = Ray();
            hits_[rayID] = Hit();
            traversal_trails_[rayID] = {};
            traversal_stacks_[rayID] = TraversalStack();
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

    void set_payload_addr(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t payload_addr = rs2_data[tid].u32;
            payload_addrs_[rayID] = payload_addr;
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

    ShaderType schedule_work(){
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

        uint32_t type = schedule_work();

        uint32_t out_warp[num_lanes_];
        uint32_t active_lanes = shader_queues[type].pop_warp(out_warp);
        
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            if(tid < active_lanes){
                uint32_t rayID = out_warp[tid];
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
                case VX_RT_RAY_PAYLOAD_ADDR: rd_data[tid].u32 = payload_addrs_[rayID]; break;

                case VX_RT_HIT_DIST: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hits_[rayID].dist); break;
                case VX_RT_HIT_BX: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hits_[rayID].bx); break;
                case VX_RT_HIT_BY: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hits_[rayID].by); break;
                case VX_RT_HIT_BZ: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hits_[rayID].bz); break;
                case VX_RT_HIT_BLAS_IDX: rd_data[tid].u32 = hits_[rayID].blasIdx; break;
                case VX_RT_HIT_TRI_IDX: rd_data[tid].u32 = hits_[rayID].triIdx; break;

                default: rd_data[tid].u32 = 0; break;
            }
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
                    rays_.erase(rayID);
                    hits_.erase(rayID);
                    traversal_trails_.erase(rayID);
                    traversal_stacks_.erase(rayID);
                    payload_addrs_.erase(rayID);
                    break;
                default: break;
            }
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

    uint32_t cur_rayid_; // 0 as the invalid ray
    std::unordered_map<uint32_t, Ray> rays_;
    std::unordered_map<uint32_t, Hit> hits_;
    std::unordered_map<uint32_t, TraversalTrail> traversal_trails_;
    std::unordered_map<uint32_t, TraversalStack> traversal_stacks_;

    std::unordered_map<uint32_t, uint32_t> payload_addrs_;

    //std::vector<ShaderQueue> shader_queues;
    std::array<ShaderQueue<SHADER_QUEUE_CAPACITY, NUM_RTU_LANES>, ShaderTypes> shader_queues;
};

RTUnit::RTUnit(const SimContext &ctx, const char* name, const Arch &arch, const DCRS &dcrs, Core* core)
    : SimObject<RTUnit>(ctx, name)
    , Inputs(ISSUE_WIDTH, this)
	, Outputs(ISSUE_WIDTH, this)
	, impl_(new Impl(this, arch, dcrs, core))
    , rtu_dcache_req_out(NUM_RTU_BLOCKS, std::vector<SimChannel<MemReq>>(NUM_RTU_LANES, this))
    , rtu_dcache_rsp_in(NUM_RTU_BLOCKS, std::vector<SimChannel<MemRsp>>(NUM_RTU_LANES, this))
{}

RTUnit::~RTUnit() {
  print_stats();
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

void RTUnit::print_stats() const {
    // PerfStats stats = perf_stats();
    // std::cout << "Total warps: " << stats.rt_total_warps << std::endl;
    // std::cout << "Utilization: " << stats.rt_active_cycles << " " <<  stats.total_elapsed_cycles << " " <<(float)stats.rt_active_cycles / stats.total_elapsed_cycles << std::endl;
    // std::cout << "Avg warp latency: " << (float)stats.rt_total_warp_latency / stats.rt_total_warps << std::endl;
    // std::cout << "Avg efficiency: " << stats.rt_total_simt_efficiency / stats.rt_total_warps << std::endl;

    // std::string warp_status_names[warp_statuses] = {
    //     "warp_stalled",
    //     "warp_waiting",
    //     "warp_executing"
    // };

    // std::string ray_status_names[ray_statuses] = {
    //     "awaiting_processing",
    //     "awaiting_scheduling",
    //     "awaiting_mf",
    //     "executing_op",
    //     "trace_complete"
    // };

    // for (unsigned i=0; i<warp_statuses; i++) {
    //     std::cout << warp_status_names[i].c_str() << std::endl;
    //     for (unsigned j=0; j<ray_statuses; j++) {
    //         std::cout << "=> " << ray_status_names[j].c_str() << ": " << stats.rt_latency_dist[i][j] / stats.rt_latency_counter << std::endl;
    //     }
    // }
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

void RTUnit::set_payload_addr(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data){
    impl_->set_payload_addr(rs1_data, rs2_data);
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

void RTUnit::commit(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, RtuTraceData* trace_data){
    impl_->commit(rs1_data, rs2_data, trace_data);
}