#include "rt_unit.h"
#include "rt_traversal.h"
#include "rt_sim.h"
#include "core.h"
#include <cassert>
#include <fstream>
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

    uint32_t init_ray(uint32_t payload_addr){
        uint32_t rayID = cur_rayid_++;
        if(rayID == 0x10000000) rayID = 1;
        rays_[rayID] = Ray();
        hits_[rayID] = Hit();
        best_hits_[rayID] = Hit();
        payload_addrs_[rayID] = payload_addr;
        traversal_trails_[rayID] = {};
        traversal_stacks_[rayID] = TraversalStack();
        return rayID;
    }

    void free_ray(uint32_t rayID){
        rays_.erase(rayID);
        hits_.erase(rayID);
        best_hits_.erase(rayID);
        payload_addrs_.erase(rayID);
        traversal_trails_.erase(rayID);
        traversal_stacks_.erase(rayID);
    }

    void traverse(uint32_t rayID, per_thread_info &thread_info){
        bool completed = bvh_traverser_.traverse(
            rays_[rayID], 
            hits_[rayID],
            best_hits_[rayID],
            traversal_trails_[rayID],
            traversal_stacks_[rayID],
            thread_info
        );
        
        if(completed){
            if(best_hits_[rayID].t == LARGE_FLOAT){
                shader_queues[ShaderType::MISS].push(payload_addrs_[rayID]);
            }else{
                shader_queues[ShaderType::CLOSET].push(payload_addrs_[rayID]);
                dcache_write(&best_hits_[rayID], payload_addrs_[rayID] + sizeof(Ray), sizeof(Hit));
                thread_info.RT_store_transactions.emplace_back(payload_addrs_[rayID], 64, StoreTransactionType::TRAVERSAL_RESULTS);
            }
            free_ray(rayID);
        }else{
            shader_queues[ShaderType::ANY].push(rayID);
        }
    }

    void traverse(const std::vector<reg_data_t>& rs1_data, RtuTraceData* trace_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t payload_addr = rs1_data[tid].u32;
            uint32_t rayID = init_ray(payload_addr);
            dcache_read(&rays_[rayID], payload_addr, sizeof(Ray));
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
                uint32_t data = out_warp[tid];
                rd_data[tid].u32 = (1 << (28 + type)) | (data & 0x0FFFFFFF); 
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

                case VX_RT_HIT_T: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hits_[rayID].t); break;
                case VX_RT_HIT_U: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hits_[rayID].u); break;
                case VX_RT_HIT_V: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hits_[rayID].v); break;
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
                case VX_RT_ANYHIT_IGNORE: 
                    traverse(rayID, trace_data->m_per_scalar_thread[tid]);
                    break;
                case VX_RT_ANYHIT_ACCEPT: 
                    best_hits_[rayID] = hits_[rayID];
                    traverse(rayID, trace_data->m_per_scalar_thread[tid]);
                    break;
                case VX_RT_INTERSECTION_IGNORE: 
                    break;
                case VX_RT_INTERSECTION_ACCEPT: 
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
    std::unordered_map<uint32_t, Hit> best_hits_;
    std::unordered_map<uint32_t, TraversalTrail> traversal_trails_;
    std::unordered_map<uint32_t, TraversalStack> traversal_stacks_;
    std::unordered_map<uint32_t, uint32_t> payload_addrs_;
    std::array<ShaderQueue<RT_SHADER_QUEUE_CAPACITY, NUM_RTU_LANES>, ShaderTypes> shader_queues;
};

RTUnit::RTUnit(const SimContext &ctx, const char* name, const Arch &arch, const DCRS &dcrs, Core* core)
    : SimObject<RTUnit>(ctx, name)
    , Inputs(ISSUE_WIDTH, this)
	, Outputs(ISSUE_WIDTH, this)
	, impl_(new Impl(this, arch, dcrs, core))
    , rtu_mem_req(NUM_RTU_BLOCKS, this)
    , rtu_mem_rsp(NUM_RTU_BLOCKS, this)
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
    PerfStats stats = perf_stats();
    std::cout << "Total warps: " << stats.rt_total_warps << std::endl;
    std::cout << "Total warps latency: " << stats.rt_total_warp_latency << std::endl;
    std::cout << "Avg warp latency: " << stats.rt_total_warp_latency / stats.rt_total_warps << std::endl;
    std::cout << "Total threads latency: " << stats.rt_total_thread_latency << std::endl;
    std::cout << "Avg threads latency: " << stats.rt_total_thread_latency / stats.rt_total_warps << std::endl;
    std::cout << "Avg efficiency: " << stats.rt_total_simt_efficiency / stats.rt_total_warps << std::endl;

    std::cout << "RT active cycles: " << stats.rt_active_cycles << std::endl;
    std::cout << "RT total cycles: " <<  stats.total_elapsed_cycles << std::endl;
    std::cout << "RT active rate: " <<  (float)stats.rt_active_cycles / stats.total_elapsed_cycles  << std::endl;

    std::string warp_status_names[warp_statuses] = {
        "warp_stalled",
        "warp_waiting",
        "warp_executing"
    };

    std::string ray_status_names[ray_statuses] = {
        "awaiting_processing",
        "awaiting_scheduling",
        "awaiting_mf",
        "executing_op",
        "trace_complete"
    };

    for (unsigned i=0; i<warp_statuses; i++) {
        std::cout << warp_status_names[i].c_str() << std::endl;
        for (unsigned j=0; j<ray_statuses; j++) {
            std::cout << "=> " << ray_status_names[j].c_str() << ": " << stats.rt_latency_dist[i][j] / stats.rt_latency_counter << std::endl;
        }
    }

    // const char * filename = "latencies.csv";
    // std::ofstream outFile(filename);

    // if (outFile.is_open()) {
    //     // Header
    //     outFile << "Warp_Latency\n";

    //     // Data
    //     for (const auto& latency : stats.rt_warp_latencies) {
    //         outFile << latency << "\n";
    //     }

    //     outFile.close();
    // } else {
    //     std::cerr << "Error: Could not open file " << filename << std::endl;
    // }
}

void RTUnit::dcache_read(void* data, uint64_t addr, uint32_t size){
    impl_->dcache_read(data, addr, size);
}

void RTUnit::dcache_write(const void* data, uint64_t addr, uint32_t size){
    impl_->dcache_write(data, addr, size);
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