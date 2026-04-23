#include "rt_unit.h"
#include "rt_traversal.h"
#include "rt_sim.h"
#include "core.h"
#include <cassert>
#include <fstream>

#define NODE_ADDR(root, idx) root + idx * sizeof(BVHNode)
#define INSTANCE_ADDR(root, idx) root + idx * sizeof(BLASNode)
#define PRIMITIVE_ADDR(root, idx) root + idx * sizeof(Triangle)

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

    void init_ray(const std::vector<reg_data_t>& rs1_data, std::vector<reg_data_t>& rd_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t payload_addr = rs1_data[tid].u32;

            uint32_t rayID = cur_rayid_++;
            if(rayID == 0x10000000) rayID = 1;
            rays_[rayID] = Ray();
            payload_addrs_[rayID] = payload_addr;
            traversal_states_[rayID] = TraversalState();
            hit_buffer_stall_counts_[rayID] = 0;
            rd_data[tid].u32 = rayID;
        }
    }

    void release_ray(const std::vector<reg_data_t>& rs1_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;

            hit_buffer_.erase(rayID);
            payload_addrs_.erase(rayID);
        }
    }

    bool isLeaf(BVHNode *node){ return node->type != BVH_INTERNAL; }
    bool isInstanceLeaf(BVHNode *node){ return node->type == INSTANCE_LEAF; }
    
    uint32_t traverse(TraversalState& state, per_thread_info &thread_info){
        uint32_t node_ptr, status;
        if(state.level == state.root_level){
            node_ptr = state.root_ptr;
            status = VX_RT_TRAVERSAL_STATUS_CONTINUE;
        }else{
            status = state.pop(node_ptr);
        }

        while(status == VX_RT_TRAVERSAL_STATUS_CONTINUE){
            BVHNode node;
            dcache_read(&node, node_ptr, sizeof(BVHNode));
            thread_info.RT_mem_accesses.emplace_back(node_ptr, sizeof(BVHNode), TransactionType::BVH_INTERNAL_NODE);

            if(!isLeaf(&node)){
                std::vector<ChildIntersection> intersections;
                ray_nBox_intersect(state.ray, node, state.best_hit.t, intersections);

                std::sort(intersections.begin(), intersections.end(), [](const ChildIntersection &a, const ChildIntersection &b) {
                    return a.dist > b.dist; //farthest ------> closest
                });

                uint32_t k = state.trail[state.level];
                uint32_t dropCount = (k == RT_BVH_WIDTH) ? intersections.size() - 1 : k;
                for(int i=0; i<dropCount; i++){
                    if(intersections.size() > 0){
                        intersections.pop_back();
                    }
                }
                
                if(intersections.size() == 0){
                    status = state.pop(node_ptr);
                }else{
                    ChildIntersection closest = intersections.back();
                    intersections.pop_back();

                    uint32_t nodeIdx = node.leftFirst + closest.childIdx;
                    node_ptr = NODE_ADDR(state.root_ptr, nodeIdx);
                    
                    if(intersections.size() == 0){
                        state.trail[state.level] = RT_BVH_WIDTH;
                    }else{
                        for(auto iter = intersections.begin(); iter != intersections.end(); iter++){
                            nodeIdx = node.leftFirst + (*iter).childIdx;
                            state.stack.push({NODE_ADDR(state.root_ptr, nodeIdx), iter == intersections.begin()});
                        }
                    }
                    state.level++;
                }

            }else{
                //Leaf Node
                if(isInstanceLeaf(&node)){
                    state.hit.instanceID = node.leaf.primCount;
                    status = VX_RT_TRAVERSAL_STATUS_INSTANCE_HIT;
                }else{
                #ifdef RT_SHADER_INTERSECTION_ENABLE
                    state.hit.primitiveID = node_ptr;
                    return VX_RT_TRAVERSAL_STATUS_TO_INTERSECTION_SHADER;
                #else
                    uint32_t leftFirst = node.leftFirst;
                    uint32_t triCount = node.leaf.primCount;

                    for (uint32_t i = 0; i < triCount; ++i) {
                        uint32_t triIdx = leftFirst + i;                    
                        uint32_t tri_addr = tri_ptr + triIdx * sizeof(Triangle);

                        Triangle tri;
                        dcache_read(&tri, tri_addr, sizeof(Triangle));
                        
                        float u, v;
                        float t = ray_tri_intersect(state.ray, tri, u, v);

                        // atomic test-and-set
                        if (t < state.best_hit.t) {

                        #ifdef RT_SHADER_ANYHIT_ENABLE
                            state.hit.t = t;
                            state.hit.u = u;
                            state.hit.v = v;
                            state.hit.primitiveID = triIdx;
                            
                            return VX_RT_TRAVERSAL_STATUS_TO_ANYHIT_SHADER;
                        #else
                            state.best_hit.t = t;
                            state.best_hit.u = u;
                            state.best_hit.v = v;
                            state.best_hit.instanceID = state.hit.instanceID;
                            state.best_hit.primitiveID = triIdx;
                        #endif
                        } 
                    }

                    thread_info.RT_mem_accesses.emplace_back(tri_ptr + leftFirst * sizeof(Triangle), 64, TransactionType::BVH_QUAD_LEAF);
                    status = state.pop(node_ptr);
                #endif
                }
            }
        }

        return status;
    }

    void traverse(uint32_t rayID, per_thread_info &thread_info){
        TraversalState& state = traversal_states_[rayID];
        
        bool exit = false;

        while(!exit){
            // Run traversal until it hits a leaf or finishes
            uint32_t status = traverse(state, thread_info);

            switch(status){
                case VX_RT_TRAVERSAL_STATUS_INSTANCE_HIT:{
                    // TLAS -> BLAS
                    BLASNode blas;
                    uint32_t instance_ptr = INSTANCE_ADDR(blas_ptr, state.hit.instanceID);
                    dcache_read(&blas, instance_ptr, sizeof(BLASNode));
                    thread_info.RT_mem_accesses.emplace_back(instance_ptr, sizeof(BLASNode), TransactionType::BVH_INSTANCE_LEAF);
                    state.ray = ray_transform(rays_[rayID], blas.invTransform);
                    state.root_ptr = NODE_ADDR(qBvh_ptr, blas.bvh_offset);
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
                    if(state.root_ptr == tlas_ptr || state.root_level == 0 || state.trail[0] == RT_BVH_WIDTH){
                        // TLAS Finished
                        if(state.best_hit.t == LARGE_FLOAT){
                            shader_queues[ShaderType::MISS].push(rayID);
                        }else{
                            shader_queues[ShaderType::CLOSET].push(rayID);
                            if(hit_buffer_.count(rayID) > 0) hit_buffer_stall_counts_[rayID]++;
                            hit_buffer_[rayID] = state.best_hit;
                        }
                        exit = true;
                    }else{
                        // BLAS Finished (BLAS -> TLAS)
                        state.ray = rays_[rayID];
                        state.root_ptr = tlas_ptr;

                        if(state.stack.empty()){
                            state.level = 0;
                        }else{
                            state.level = state.root_level;
                        }

                        state.root_level = 0;
                    }
                    break;
                }

                default: std::abort();
            }

        }
    }

    void traverse(const std::vector<reg_data_t>& rs1_data, RtuTraceData* trace_data){
        tlas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TLAS_PTR);
        blas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BLAS_PTR);
        qBvh_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BVH_PTR);
        tri_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_PTR);

        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            traversal_states_[rayID] = TraversalState(rays_[rayID], tlas_ptr);
            traverse(rayID, trace_data->m_per_scalar_thread[tid]);
        }
    }

    ShaderType schedule_work(){
        ShaderType targetType = ShaderType::MISS;
        if(shader_queues[ShaderType::CLOSET].size() > shader_queues[targetType].size()){
            targetType = ShaderType::CLOSET;
        }  

        if(shader_queues[ShaderType::INTERSECTION].size() > shader_queues[targetType].size()){
            targetType = ShaderType::INTERSECTION;
        }

        if(shader_queues[ShaderType::ANY].size() > shader_queues[targetType].size()){
            targetType = ShaderType::ANY;
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
            if(rayID == 0) continue;

            uint32_t attrID = rs2_data[tid].u32;

            TraversalState& state = traversal_states_[rayID];

            switch(attrID){
                case VX_RT_WORLD_RAY_RO_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&rays_[rayID].ro_x); break;
                case VX_RT_WORLD_RAY_RO_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&rays_[rayID].ro_y); break;
                case VX_RT_WORLD_RAY_RO_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&rays_[rayID].ro_z); break;
                case VX_RT_WORLD_RAY_RD_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&rays_[rayID].rd_x); break;
                case VX_RT_WORLD_RAY_RD_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&rays_[rayID].rd_y); break;
                case VX_RT_WORLD_RAY_RD_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&rays_[rayID].rd_z); break;
                
                case VX_RT_OBJECT_RAY_RO_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.ray.ro_x); break;
                case VX_RT_OBJECT_RAY_RO_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.ray.ro_y); break;
                case VX_RT_OBJECT_RAY_RO_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.ray.ro_z); break;
                case VX_RT_OBJECT_RAY_RD_X: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.ray.rd_x); break;
                case VX_RT_OBJECT_RAY_RD_Y: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.ray.rd_y); break;
                case VX_RT_OBJECT_RAY_RD_Z: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.ray.rd_z); break;

                case VX_RT_HIT_T: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.hit.t); break;
                case VX_RT_HIT_U: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.hit.u); break;
                case VX_RT_HIT_V: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.hit.v); break;
                case VX_RT_HIT_INSTANCE_ID: rd_data[tid].u32 = state.hit.instanceID; break;
                case VX_RT_HIT_PRIMITIVE_ID: rd_data[tid].u32 = state.hit.primitiveID; break;

                // Hit results are only for CHS
                case VX_RT_HIT_RESULT_T: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hit_buffer_[rayID].t); break;
                case VX_RT_HIT_RESULT_U: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hit_buffer_[rayID].u); break;
                case VX_RT_HIT_RESULT_V: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&hit_buffer_[rayID].v); break;
                case VX_RT_HIT_RESULT_INSTANCE_ID: rd_data[tid].u32 = hit_buffer_[rayID].instanceID; break;
                case VX_RT_HIT_RESULT_PRIMITIVE_ID: rd_data[tid].u32 = hit_buffer_[rayID].primitiveID; break;

                case VX_RT_PAYLOAD_ADDR: rd_data[tid].u32 = payload_addrs_[rayID]; break;
                default: rd_data[tid].u32 = 0; break;
            }
        } 
    }

    void set_attr(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data, uint32_t attr){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            if(rayID == 0) continue;

            uint32_t rs2 = rs2_data[tid].u32;
            uint32_t rs3 = rs3_data[tid].u32;
            
            float v0 = *reinterpret_cast<float*>(&rs2);
            float v1 = *reinterpret_cast<float*>(&rs3);

            switch(attr){
                case VX_RT_RAY_X:
                    rays_[rayID].ro_x = v0;
                    rays_[rayID].rd_x = v1;
                    break;
                case VX_RT_RAY_Y:
                    rays_[rayID].ro_y = v0;
                    rays_[rayID].rd_y = v1;
                    break;
                case VX_RT_RAY_Z:
                    rays_[rayID].ro_z = v0;
                    rays_[rayID].rd_z = v1;
                    break;
                case VX_RT_HIT_T_ID:
                    traversal_states_[rayID].hit.t = v0;
                    traversal_states_[rayID].hit.primitiveID = rs3;
                    break;
                case VX_RT_HIT_UV:
                    traversal_states_[rayID].hit.u = v0;
                    traversal_states_[rayID].hit.v = v1;
                    break;
                default: break;
            }
        }  
    }

    void commit(const std::vector<reg_data_t>& rs1_data, uint32_t action, RtuTraceData* trace_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            if(rayID == 0) continue;

            switch(action){
                case VX_RT_ANYHIT_IGNORE: 
                case VX_RT_INTERSECTION_IGNORE:
                    traverse(rayID, trace_data->m_per_scalar_thread[tid]);
                    break;
                case VX_RT_ANYHIT_ACCEPT: 
                    traversal_states_[rayID].best_hit = traversal_states_[rayID].hit;
                    traverse(rayID, trace_data->m_per_scalar_thread[tid]);
                    break;
                case VX_RT_INTERSECTION_ACCEPT: {
                    #ifdef RT_SHADER_ANYHIT_ENABLE
                        if(traversal_states_[rayID].hit.t < traversal_states_[rayID].best_hit.t){
                            shader_queues[ShaderType::ANY].push(rayID);
                        }else{
                            traverse(rayID, trace_data->m_per_scalar_thread[tid]);
                        }
                    #else
                        if(traversal_states_[rayID].hit.t < traversal_states_[rayID].best_hit.t){
                            traversal_states_[rayID].best_hit = traversal_states_[rayID].hit;
                        }
                        traverse(rayID, trace_data->m_per_scalar_thread[tid]);
                    #endif
                    break;
                }
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

    uint32_t tlas_ptr, blas_ptr, qBvh_ptr, tri_ptr;

    uint32_t cur_rayid_; // 0 as the invalid ray
    std::unordered_map<uint32_t, Ray> rays_;
    std::unordered_map<uint32_t, Hit> hit_buffer_; // Stored in specialized SRAM in a RT Core
    std::unordered_map<uint32_t, uint32_t> payload_addrs_;
    std::unordered_map<uint32_t, TraversalState> traversal_states_; // Stored in RT Core Latches/Registers
    std::array<ShaderQueue<RT_SHADER_QUEUE_CAPACITY, NUM_RTU_LANES>, ShaderTypes> shader_queues;

    std::unordered_map<uint32_t, uint32_t> hit_buffer_stall_counts_;
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

void RTUnit::init_ray(const std::vector<reg_data_t>& rs1_data, std::vector<reg_data_t>& rd_data){
    impl_->init_ray(rs1_data, rd_data);
}

void RTUnit::set_attr(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, const std::vector<reg_data_t>& rs3_data, uint32_t attr){
    impl_->set_attr(rs1_data, rs2_data, rs3_data, attr);
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

void RTUnit::commit(const std::vector<reg_data_t>& rs1_data, uint32_t action, RtuTraceData* trace_data){
    impl_->commit(rs1_data, action, trace_data);
}

void RTUnit::release_ray(const std::vector<reg_data_t>& rs1_data){
    impl_->release_ray(rs1_data);
}