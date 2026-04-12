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
        payload_addrs_[rayID] = payload_addr;
        return rayID;
    }

    void free_ray(uint32_t rayID){
        rays_.erase(rayID);
        traversal_states_.erase(rayID);
        payload_addrs_.erase(rayID);
    }

    void traverse(uint32_t rayID, per_thread_info &thread_info){
        uint32_t tlas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TLAS_PTR);
        uint32_t blas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BLAS_PTR);
        uint32_t qBvh_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BVH_PTR);
       
        TraversalState& state = traversal_states_[rayID];
        
        bool exit = false;

        while(!exit){
            // Run traversal until it hits a leaf or finishes
            uint32_t status = bvh_traverser_.traverse(state, thread_info);

            switch(status){
                case VX_RT_TRAVERSAL_STATUS_TLAS_LEAF_HIT:{
                    // TLAS -> BLAS
                    BLASNode blas;
                    dcache_read(&blas, blas_ptr + (state.hit.blasIdx) * 160, sizeof(BLASNode));
                    state.ray = bvh_traverser_.ray_transform(rays_[rayID], blas.invTransform);
                    state.root_ptr = qBvh_ptr + blas.bvh_offset * sizeof(BVHNode);
                    state.root_level = state.level;
                    break;
                }

                case VX_RT_TRAVERSAL_STATUS_RESTART: {
                    state.level = state.root_level;
                    break;
                }

               case VX_RT_TRAVERSAL_STATUS_TO_ANYHIT_SHADER: {
                    shader_queues[ShaderType::ANY].push(rayID);
                    exit = true;
                    break;
               }

               case VX_RT_TRAVERSAL_STATUS_TO_INTERSECTION_SHADER: {
                    shader_queues[ShaderType::INTERSECTION].push(rayID);
                    exit = true;
                    break;
               }

                case VX_RT_TRAVERSAL_STATUS_FINISHED: {
                    if(state.root_ptr == tlas_ptr || state.trail[0] == RT_BVH_WIDTH){
                        // TLAS Finished
                        if(state.best_hit.t == LARGE_FLOAT){
                            shader_queues[ShaderType::MISS].push(payload_addrs_[rayID]);
                        }else{
                            shader_queues[ShaderType::CLOSET].push(payload_addrs_[rayID]);
                            dcache_write(&state.best_hit, payload_addrs_[rayID] + sizeof(Ray), sizeof(Hit));
                            thread_info.RT_store_transactions.emplace_back(payload_addrs_[rayID], 64, StoreTransactionType::TRAVERSAL_RESULTS);
                        }
                        free_ray(rayID);
                        exit = true;
                    }else{
                        // BLAS Finished (BLAS -> TLAS)
                        state.ray = rays_[rayID];
                        state.root_ptr = tlas_ptr;
                        state.root_level = 0;

                        if(state.stack.empty()){
                            state.level = 0;
                        }else{
                            state.level = state.root_level;
                        }
                    }
                    break;
                }

                default: break;
            }

        }
    }

    void traverse(const std::vector<reg_data_t>& rs1_data, RtuTraceData* trace_data){
        uint32_t tlas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TLAS_PTR);

        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t payload_addr = rs1_data[tid].u32;
            uint32_t rayID = init_ray(payload_addr);
            dcache_read(&rays_[rayID], payload_addr, sizeof(Ray));
            traversal_states_[rayID] = TraversalState(rays_[rayID], tlas_ptr);
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

        if(shader_queues[ShaderType::INTERSECTION].size() > shader_queues[targetType].size()){
            targetType = ShaderType::INTERSECTION;
        }

        return targetType;
    }

    void get_work(std::vector<reg_data_t>& rd_data){
        if(shader_queues[ShaderType::MISS].empty() && shader_queues[ShaderType::CLOSET].empty() && 
        shader_queues[ShaderType::ANY].empty() && shader_queues[ShaderType::INTERSECTION].empty()){
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

            TraversalState& state = traversal_states_[rayID];

            switch(attrID){
                case VX_RT_RAY_RO_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.ray.ro_x); break;
                case VX_RT_RAY_RO_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.ray.ro_y); break;
                case VX_RT_RAY_RO_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.ray.ro_z); break;
                case VX_RT_RAY_RD_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.ray.rd_x); break;
                case VX_RT_RAY_RD_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.ray.rd_y); break;
                case VX_RT_RAY_RD_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.ray.rd_z); break;

                case VX_RT_HIT_T: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.hit.t); break;
                case VX_RT_HIT_U: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.hit.u); break;
                case VX_RT_HIT_V: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.hit.v); break;
                case VX_RT_HIT_BLAS_IDX: rd_data[tid].u32 = state.hit.blasIdx; break;
                case VX_RT_HIT_TRI_IDX: rd_data[tid].u32 = state.hit.triIdx; break;

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
                    traversal_states_[rayID].best_hit = traversal_states_[rayID].hit;
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
    std::unordered_map<uint32_t, TraversalState> traversal_states_;
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
    // PerfStats stats = perf_stats();
    // std::cout << "Total warps: " << stats.rt_total_warps << std::endl;
    // std::cout << "Total warps latency: " << stats.rt_total_warp_latency << std::endl;
    // std::cout << "Avg warp latency: " << stats.rt_total_warp_latency / stats.rt_total_warps << std::endl;
    // std::cout << "Total threads latency: " << stats.rt_total_thread_latency << std::endl;
    // std::cout << "Avg threads latency: " << stats.rt_total_thread_latency / stats.rt_total_warps << std::endl;
    // std::cout << "Avg efficiency: " << stats.rt_total_simt_efficiency / stats.rt_total_warps << std::endl;

    // std::cout << "RT active cycles: " << stats.rt_active_cycles << std::endl;
    // std::cout << "RT total cycles: " <<  stats.total_elapsed_cycles << std::endl;
    // std::cout << "RT active rate: " <<  (float)stats.rt_active_cycles / stats.total_elapsed_cycles  << std::endl;

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