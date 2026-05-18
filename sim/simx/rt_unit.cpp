#include "rt_unit.h"
#include "rt_sim.h"
#include "rt_core.h"
#include "core.h"
#include <cassert>
#include <fstream>

using namespace vortex;

class RTUnit::Impl {
public:
    Impl(RTUnit* simobject, const Arch &arch, const DCRS &dcrs, Core* core)
        : simobject_(simobject)
        , rt_core_(new RTCore(simobject, dcrs))
        , rt_sim_(new RTSim(simobject))
        , core_(core)
        , arch_(arch)
        , dcrs_(dcrs)
        , num_blocks_(NUM_RTU_BLOCKS)
        , num_lanes_(NUM_RTU_LANES)
        , ray_id_(0)
        , shader_states_(arch.num_warps(), std::vector<ShaderState>(arch.num_threads()))
    {}

    ~Impl() {
        delete rt_core_;
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

    void traverse(uint32_t wid, uint32_t tid, const std::vector<reg_data_t>& rs1_data, RtuTraceData* trace_data){
        ShaderState& state = shader_states_.at(wid).at(tid);

        uint32_t tlas_addr = rs1_data[tid].u32;

        uint32_t rayID = (ray_id_++) & 0x0FFFFFFF;

        payload_addrs_[rayID] = state.payload_addr;

        rt_core_->allocate(rayID, state.world_ray, tlas_addr, state.tmin, state.tmax);
        rt_core_->traverse(rayID, trace_data->m_per_scalar_thread[tid]);
    }

    void get_work(uint32_t wid, std::vector<reg_data_t>& rd_data){
        uint32_t active_lanes;
        uint32_t out_warp[SIMD_WIDTH];
        ShaderType type = rt_core_->shader_queue_pop(out_warp, active_lanes);

        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            if(tid < active_lanes){
                uint32_t data = out_warp[tid];
                uint32_t rayID = data & 0x0FFFFFFF;
                uint32_t hitID = data >> 28;

                const Ray& world_ray = rt_core_->get_world_ray(rayID);
                const TraversalState& state = rt_core_->get_traversal_state(rayID);

                switch(type){
                    case ShaderType::MISS:{
                        shader_states_.at(wid).at(tid).world_ray = world_ray;
                        shader_states_.at(wid).at(tid).tmin = state.tmin;
                        shader_states_.at(wid).at(tid).tmax = state.best_hit.t;
                        shader_states_.at(wid).at(tid).payload_addr = payload_addrs_[rayID];
                        break;
                    }

                    case ShaderType::CLOSET:{
                        shader_states_.at(wid).at(tid).world_ray = world_ray;
                        shader_states_.at(wid).at(tid).object_ray = state.ray;
                        shader_states_.at(wid).at(tid).tmin = state.tmin;
                        shader_states_.at(wid).at(tid).tmax = state.best_hit.t;
                        shader_states_.at(wid).at(tid).hit = state.best_hit;
                        shader_states_.at(wid).at(tid).payload_addr = payload_addrs_[rayID];
                        break;
                    }

                    case ShaderType::ANYHIT:{
                        shader_states_.at(wid).at(tid).id = data;
                        shader_states_.at(wid).at(tid).world_ray = world_ray;
                        shader_states_.at(wid).at(tid).object_ray = state.ray;
                        shader_states_.at(wid).at(tid).tmin = state.tmin;
                        shader_states_.at(wid).at(tid).tmax = state.best_hit.t;
                        shader_states_.at(wid).at(tid).hit = state.prim_hit[hitID];
                        shader_states_.at(wid).at(tid).payload_addr = payload_addrs_[rayID];
                        break;
                    }

                    case ShaderType::INTERSECTION:{
                        shader_states_.at(wid).at(tid).id = data;
                        shader_states_.at(wid).at(tid).world_ray = world_ray;
                        shader_states_.at(wid).at(tid).object_ray = state.ray;
                        shader_states_.at(wid).at(tid).tmin = state.tmin;
                        shader_states_.at(wid).at(tid).tmax = state.best_hit.t;
                        shader_states_.at(wid).at(tid).hit = state.prim_hit[hitID];
                        shader_states_.at(wid).at(tid).payload_addr = payload_addrs_[rayID];
                        break;
                    }

                    default: std::abort();
                }

                rd_data[tid].u32 = (1 << type); 
            }else{
                rd_data[tid].u32 = 0; 
            }
        }
    }

    void get_attr(uint32_t wid, uint32_t tid, uint32_t attr, std::vector<reg_data_t>& rd_data){
        ShaderState& state = shader_states_.at(wid).at(tid);

        switch(attr){
            case VX_RT_WORLD_RAY_RO_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.world_ray.ro_x); break;
            case VX_RT_WORLD_RAY_RO_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.world_ray.ro_y); break;
            case VX_RT_WORLD_RAY_RO_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.world_ray.ro_z); break;
            case VX_RT_WORLD_RAY_RD_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.world_ray.rd_x); break;
            case VX_RT_WORLD_RAY_RD_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.world_ray.rd_y); break;
            case VX_RT_WORLD_RAY_RD_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.world_ray.rd_z); break;
            
            case VX_RT_OBJECT_RAY_RO_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.object_ray.ro_x); break;
            case VX_RT_OBJECT_RAY_RO_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.object_ray.ro_y); break;
            case VX_RT_OBJECT_RAY_RO_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.object_ray.ro_z); break;
            case VX_RT_OBJECT_RAY_RD_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.object_ray.rd_x); break;
            case VX_RT_OBJECT_RAY_RD_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.object_ray.rd_y); break;
            case VX_RT_OBJECT_RAY_RD_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.object_ray.rd_z); break;

            case VX_RT_T_MIN: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.tmin); break;
            case VX_RT_T_MAX: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.tmax); break;
            
            case VX_RT_HIT_T: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.hit.t); break;
            case VX_RT_HIT_ATTR_0: rd_data[tid].u32 = state.hit.attrs[0]; break;
            case VX_RT_HIT_ATTR_1: rd_data[tid].u32 = state.hit.attrs[1]; break;
            case VX_RT_HIT_ATTR_2: rd_data[tid].u32 = state.hit.attrs[2]; break;
            case VX_RT_HIT_ATTR_3: rd_data[tid].u32 = state.hit.attrs[3]; break;
            case VX_RT_HIT_ATTR_4: rd_data[tid].u32 = state.hit.attrs[4]; break;
            case VX_RT_HIT_ATTR_5: rd_data[tid].u32 = state.hit.attrs[5]; break;
            case VX_RT_HIT_ATTR_6: rd_data[tid].u32 = state.hit.attrs[6]; break;
            case VX_RT_HIT_ATTR_7: rd_data[tid].u32 = state.hit.attrs[7]; break;
            case VX_RT_HIT_INSTANCE_ID: rd_data[tid].u32 = state.hit.instanceID; break;
            case VX_RT_HIT_PRIMITIVE_ID: rd_data[tid].u32 = state.hit.primitiveID; break;
            case VX_RT_HIT_GEOMETRY_INDEX: rd_data[tid].u32 = state.hit.geometryIndex; break;
            
            case VX_RT_PAYLOAD_ADDR: rd_data[tid].u32 = state.payload_addr; break;
            default: rd_data[tid].u32 = 0; break;
        }
    }

    void set_attr(uint32_t wid, uint32_t tid, uint32_t attr, const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data){
        ShaderState& state = shader_states_.at(wid).at(tid);

        uint32_t u1 = rs1_data[tid].u32;
        uint32_t u2 = rs2_data[tid].u32;
        uint32_t u3 = rs3_data[tid].u32;
        
        float f1 = *reinterpret_cast<float*>(&u1);
        float f2 = *reinterpret_cast<float*>(&u2);
        float f3 = *reinterpret_cast<float*>(&u3);
        
        switch(attr){
            case VX_RT_RAY_ORIGIN:
                state.world_ray.ro_x = f1;
                state.world_ray.ro_y = f2;
                state.world_ray.ro_z = f3;
                break;
            case VX_RT_RAY_DIRECTION:
                state.world_ray.rd_x = f1;
                state.world_ray.rd_y = f2;
                state.world_ray.rd_z = f3;
                break;
            case VX_RT_RAY_T_PAYLOAD:
                state.tmin = f1;
                state.tmax = f2;
                state.payload_addr = u3;
                break;
            case VX_RT_HIT_ATTR_0: state.hit.attrs[0] = u1; break;
            case VX_RT_HIT_ATTR_1: state.hit.attrs[1] = u1; break;
            case VX_RT_HIT_ATTR_2: state.hit.attrs[2] = u1; break;
            case VX_RT_HIT_ATTR_3: state.hit.attrs[3] = u1; break;
            case VX_RT_HIT_ATTR_4: state.hit.attrs[4] = u1; break;
            case VX_RT_HIT_ATTR_5: state.hit.attrs[5] = u1; break;
            case VX_RT_HIT_ATTR_6: state.hit.attrs[6] = u1; break;
            case VX_RT_HIT_ATTR_7: state.hit.attrs[7] = u1; break;
            case VX_RT_HIT_T: state.hit.t = f1; break; // experimental
            default: break;
        }
    }

    void commit(uint32_t wid, uint32_t tid, uint32_t action, RtuTraceData* trace_data){
        ShaderState& state = shader_states_.at(wid).at(tid);

        uint32_t rayID = state.id & 0x0FFFFFFF;
        uint32_t hitID = state.id >> 28;

        switch(action){
            case VX_RT_ANYHIT_IGNORE: 
                rt_core_->commit(rayID, hitID, Hit(), ShaderType::ANYHIT, trace_data->m_per_scalar_thread[tid]);
                break;
            case VX_RT_INTERSECTION_IGNORE:
                 rt_core_->commit(rayID, hitID, Hit(), ShaderType::INTERSECTION, trace_data->m_per_scalar_thread[tid]);
                break;
            case VX_RT_ANYHIT_ACCEPT: 
                rt_core_->commit(rayID, hitID, state.hit, ShaderType::ANYHIT, trace_data->m_per_scalar_thread[tid]);
                break;
            case VX_RT_INTERSECTION_ACCEPT:
                rt_core_->commit(rayID, hitID, state.hit, ShaderType::INTERSECTION, trace_data->m_per_scalar_thread[tid]);
                break;
            default: break;
        } 
    }
    
private:
    RTUnit*       simobject_;
    RTCore*       rt_core_;
    RTSim*        rt_sim_;
    Core*         core_;
    const Arch&   arch_;
    const DCRS&   dcrs_;

    uint32_t num_blocks_;
    uint32_t num_lanes_;

    uint32_t ray_id_;
    std::unordered_map<uint32_t, uint32_t> payload_addrs_;

    struct ShaderState {
        uint32_t id;
        Ray world_ray;
        Ray object_ray;
        Hit hit;
        float tmax;
        float tmin;
        uint32_t payload_addr;
    };

    std::vector<std::vector<ShaderState>> shader_states_;
};

RTUnit::RTUnit(const SimContext &ctx, const char* name, const Arch &arch, const DCRS &dcrs, Core* core)
    : SimObject<RTUnit>(ctx, name)
    , Inputs(ISSUE_WIDTH, this)
	, Outputs(ISSUE_WIDTH, this)
    , rtu_mem_req(NUM_RTU_BLOCKS, this)
    , rtu_mem_rsp(NUM_RTU_BLOCKS, this)
    , impl_(new Impl(this, arch, dcrs, core))
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

void RTUnit::traverse(uint32_t wid, uint32_t tid, const std::vector<reg_data_t>& rs1_data, RtuTraceData* trace_data){
    impl_->traverse(wid, tid, rs1_data, trace_data);
}

void RTUnit::get_work(uint32_t wid, std::vector<reg_data_t>& rd_data){
    impl_->get_work(wid, rd_data);
}

void RTUnit::get_attr(uint32_t wid, uint32_t tid, uint32_t attr, std::vector<reg_data_t>& rd_data){
    impl_->get_attr(wid, tid, attr, rd_data);
}

void RTUnit::set_attr(uint32_t wid, uint32_t tid, uint32_t attr, const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data){
    impl_->set_attr(wid, tid, attr, rs1_data, rs2_data, rs3_data);
}

void RTUnit::commit(uint32_t wid, uint32_t tid, uint32_t action, RtuTraceData* trace_data){
    impl_->commit(wid, tid, action, trace_data);
}

// void RTUnit::release_ray(const std::vector<reg_data_t>& rs1_data){
//     impl_->release_ray(rs1_data);
// }