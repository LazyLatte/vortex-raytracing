#include "rt_unit.h"
#include "rt_traversal.h"
#include "rt_sim.h"
#include "core.h"
#include <cassert>
#include <fstream>

#define NODE_ADDR(root, idx) root + (idx) * sizeof(BVHNode)
#define INSTANCE_ADDR(root, idx) root + (idx) * sizeof(BLASNode)
#define TRI_ADDR(root, idx) root + (idx) * sizeof(Triangle)
#define PRIM_AABB_ADDR(root, idx) root + (idx) * sizeof(AABB)

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
    bool isProceduralLeaf(BVHNode *node){ return node->type == PROCEDURAL_LEAF; }

    void traverse(TraversalState& state, per_thread_info &thread_info){
        uint32_t node_ptr = (state.level == state.root_level) ? state.root_ptr : state.pop();

        while(state.status == TraversalStatus::CONTINUE){
            BVHNode node;
            dcache_read(&node, node_ptr, sizeof(BVHNode));
            thread_info.RT_mem_accesses.emplace_back(node_ptr, sizeof(BVHNode), TransactionType::BVH_INTERNAL_NODE);

            if(!isLeaf(&node)){
                std::array<BoxHit, RT_BOX_INTERSECTION_WIDTH> box_hits;
                uint32_t valid_count = ray_nBox_intersect(node, state, box_hits); // SIMD intersection

                uint32_t k = state.trail[state.level];
                uint32_t start = (k == RT_BVH_WIDTH) ? valid_count - 1 : k;
                uint32_t end = valid_count;

                if(valid_count == 0 || start >= end){
                    node_ptr = state.pop();
                }else{
                    std::sort(box_hits.begin(), box_hits.end(), BoxHit::compare);

                    BoxHit closest = box_hits[start++];
                    node_ptr = NODE_ADDR(state.root_ptr, node.leftFirst + closest.idx);
                    
                    if(start == end){
                        state.trail[state.level] = RT_BVH_WIDTH;
                    }else{
                        for (int32_t i = (int32_t)end - 1; i >= (int32_t)start; i--) {
                            uint32_t node_addr = NODE_ADDR(state.root_ptr, node.leftFirst + box_hits[i].idx);
                            bool isFarthest = (i == end - 1);
                            state.stack.push(node_addr | isFarthest); // encode one bit info into addr
                        }
                    }
                    state.level++;
                }
            }else{
                //Leaf Node
                if(isInstanceLeaf(&node)){
                    state.instanceID = node.leaf.instanceID;
                    state.status = TraversalStatus::INSTANCE_HIT;
                }else{  
                    state.prim_count = node.leaf.prim_count;
                    state.prim_base_id = node.leftFirst;
                    state.prim_batch_finished_count = 0;
                    state.leaf_flags = node.leaf.flags;
                    state.status = isProceduralLeaf(&node) ? TraversalStatus::PROCEDURAL_LEAF_HIT : TraversalStatus::TRI_LEAF_HIT;
                }
            }
        }
    }

    void traverse(uint32_t rayID, per_thread_info &thread_info){
        TraversalState& state = traversal_states_[rayID];
        
        while(1){
            switch(state.status){
                case TraversalStatus::CONTINUE: {
                    traverse(state, thread_info);
                    break;
                }

                case TraversalStatus::INSTANCE_HIT:{
                    // TLAS -> BLAS
                    BLASNode blas;
                    uint32_t instance_ptr = INSTANCE_ADDR(blas_ptr, state.instanceID);
                    dcache_read(&blas, instance_ptr, 52);
                    thread_info.RT_mem_accesses.emplace_back(instance_ptr, 52, TransactionType::BVH_INSTANCE_LEAF);
                    state.ray = ray_transform(rays_[rayID], blas.invTransform);
                    state.root_ptr = NODE_ADDR(bvh_ptr, blas.bvh_offset);
                    state.root_level = state.level;
                    state.status = TraversalStatus::CONTINUE;
                    break;
                }

                case TraversalStatus::TRI_LEAF_HIT: {
                    uint32_t tri_count = state.prim_count;
                    uint32_t tri_base_id = state.prim_base_id;
                    uint32_t tri_finished_count = state.prim_batch_finished_count * RT_TRI_INTERSECTION_WIDTH;

                    Triangle tris[RT_TRI_INTERSECTION_WIDTH];

                    for(uint32_t i = tri_finished_count; i < tri_count; i += RT_TRI_INTERSECTION_WIDTH){
                        uint32_t tri_start_id = tri_base_id + i; 
                        uint32_t tri_start_addr = TRI_ADDR(tri_ptr, tri_start_id);

                        uint32_t tri_batch_size = std::min(tri_count - i, (uint32_t)RT_TRI_INTERSECTION_WIDTH);
                        uint32_t tri_fetch_size = sizeof(Triangle) * tri_batch_size;

                        dcache_read(&tris[0], tri_start_addr, tri_fetch_size);
                        thread_info.RT_mem_accesses.emplace_back(tri_start_addr, tri_fetch_size, TransactionType::BVH_QUAD_LEAF);

                        ray_nTri_intersect(tris, tri_start_id, tri_batch_size, state);

                        if(state.has_prim_hit()){

                            if(state.leaf_flags == NON_OPAQUE){
                                for(uint32_t idx = 0; idx < RT_TRI_INTERSECTION_WIDTH; idx++){
                                    if(state.prim_hit[idx].valid){
                                        shader_queues[ShaderType::ANY].push((idx << 24) | rayID);
                                    }
                                }
                                return;
                            }

                            else if(state.leaf_flags == OPAQUE){
                                // Comparator Tree
                                // Assume RT_TRI_INTERSECTION_WIDTH == 4 for now
                                Hit h0 = Hit::compare(state.prim_hit[0], state.prim_hit[1]);
                                Hit h1 = Hit::compare(state.prim_hit[2], state.prim_hit[3]);
                                state.best_hit = Hit::compare(h0, h1); 
                            }

                        }
                    }

                    state.status = TraversalStatus::CONTINUE;
                    break;
                }

                case TraversalStatus::PROCEDURAL_LEAF_HIT: {
                    uint32_t prim_count = state.prim_count;
                    uint32_t prim_base_id = state.prim_base_id;
                    uint32_t prim_finished_count = state.prim_batch_finished_count * RT_BOX_INTERSECTION_WIDTH;

                    AABB prim_boxes[RT_BOX_INTERSECTION_WIDTH];

                    for(uint32_t i = prim_finished_count; i < prim_count; i += RT_BOX_INTERSECTION_WIDTH){
                        uint32_t prim_start_id = prim_base_id + i; 
                        uint32_t prim_aabb_start_addr = PRIM_AABB_ADDR(aabb_ptr, prim_start_id);

                        uint32_t prim_batch_size = std::min(prim_count - i, (uint32_t)RT_BOX_INTERSECTION_WIDTH);
                        uint32_t prim_fetch_size = sizeof(AABB) * prim_batch_size;

                        dcache_read(&prim_boxes[0], prim_aabb_start_addr, prim_fetch_size);
                        thread_info.RT_mem_accesses.emplace_back(prim_aabb_start_addr, prim_fetch_size, TransactionType::BVH_PROCEDURAL_LEAF);

                        ray_nBox_intersect(prim_boxes, prim_start_id, prim_batch_size, state); 

                        if(state.has_prim_hit()){
                            for(uint32_t idx = 0; idx < RT_BOX_INTERSECTION_WIDTH; idx++){
                                if(state.prim_hit[idx].valid){
                                    shader_queues[ShaderType::INTERSECTION].push((idx << 24) | rayID);
                                }
                            }
                            return;
                        }
                    }

                    state.status = TraversalStatus::CONTINUE;
                    break;
                }

                case TraversalStatus::RESTART: {
                    state.level = state.root_level;
                    state.status = TraversalStatus::CONTINUE;
                    break;
                }

                case TraversalStatus::FINISHED: {
                    if(state.root_ptr == tlas_ptr || state.root_level == 0 || state.trail[0] == RT_BVH_WIDTH){
                        // TLAS Finished
                        if(state.best_hit.valid){
                            shader_queues[ShaderType::CLOSET].push(rayID);
                            if(hit_buffer_.count(rayID) > 0) hit_buffer_stall_counts_[rayID]++;
                            hit_buffer_[rayID] = state.best_hit;
                        }else{
                            shader_queues[ShaderType::MISS].push(rayID);
                        }
                        return;
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

                        state.status = TraversalStatus::CONTINUE;
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
        bvh_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BVH_PTR);
        tri_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_PTR);
        aabb_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_AABB_PTR);

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
                uint32_t data = out_warp[tid];
                rd_data[tid].u32 = (1 << (28 + type)) | (data & 0x0FFFFFFF); 
            }else{
                rd_data[tid].u32 = (1 << (28 + type)); 
            }
        }
    }

    void get_attr(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, std::vector<reg_data_t>& rd_data, uint32_t attr){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t hitID = rs2_data[tid].u32;

            if(rayID == 0) continue;

            TraversalState& state = traversal_states_[rayID];

            switch(attr){
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

                case VX_RT_HIT_T: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.prim_hit[hitID].t); break;
                case VX_RT_HIT_U: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.prim_hit[hitID].u); break;
                case VX_RT_HIT_V: rd_data[tid].u32 = *reinterpret_cast<uint32_t*>(&state.prim_hit[hitID].v); break;
                case VX_RT_HIT_INSTANCE_ID: rd_data[tid].u32 = state.prim_hit[hitID].instanceID; break;
                case VX_RT_HIT_PRIMITIVE_ID: rd_data[tid].u32 = state.prim_hit[hitID].primitiveID; break;

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
            uint32_t rs2 = rs2_data[tid].u32;
            uint32_t rs3 = rs3_data[tid].u32;
            uint32_t hitID = rs2;

            if(rayID == 0) continue;

            float v0 = *reinterpret_cast<float*>(&rs2);
            float v1 = *reinterpret_cast<float*>(&rs3);
            
            TraversalState& state = traversal_states_[rayID];

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
                case VX_RT_RAY_T:
                    state.tmin = v0;
                    state.best_hit.t = v1;
                    break;
                case VX_RT_HIT_T:
                    state.prim_hit[hitID].t = v1;
                    break;
                case VX_RT_HIT_U:
                    state.prim_hit[hitID].u = v1;
                    break;
                case VX_RT_HIT_V:
                    state.prim_hit[hitID].v = v1;
                    break;
                default: break;
            }
        }  
    }

    void commit(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, uint32_t action, RtuTraceData* trace_data){
        for (uint32_t tid = 0; tid < num_lanes_; tid++) {
            uint32_t rayID = rs1_data[tid].u32;
            uint32_t hitID = rs2_data[tid].u32;
            
            if(rayID == 0 || hitID >= RT_BOX_INTERSECTION_WIDTH) continue;
            
            TraversalState& state = traversal_states_[rayID];
            
            switch(action){
                case VX_RT_ANYHIT_IGNORE: 
                case VX_RT_INTERSECTION_IGNORE:
                    state.prim_hit[hitID].valid = false;
                    
                    if(!state.has_prim_hit()){
                        state.prim_batch_finished_count++;
                        traverse(rayID, trace_data->m_per_scalar_thread[tid]);
                    }

                    break;
                case VX_RT_ANYHIT_ACCEPT: 
                    if(state.prim_hit[hitID].t < state.best_hit.t){
                        state.best_hit = state.prim_hit[hitID];
                    }

                    state.prim_hit[hitID].valid = false;

                    if(!state.has_prim_hit()){
                        state.prim_batch_finished_count++;
                        traverse(rayID, trace_data->m_per_scalar_thread[tid]);
                    }

                    break;
                case VX_RT_INTERSECTION_ACCEPT: {
                    if(state.leaf_flags == OPAQUE){
                        // No AHS
                        if(state.prim_hit[hitID].t < state.best_hit.t){
                            state.best_hit = state.prim_hit[hitID];
                        }

                        state.prim_hit[hitID].valid = false;

                        if(!state.has_prim_hit()){
                            state.prim_batch_finished_count++;
                            traverse(rayID, trace_data->m_per_scalar_thread[tid]);
                        }
                    }else{
                        if(state.prim_hit[hitID].t < state.best_hit.t){
                            // candidate
                            shader_queues[ShaderType::ANY].push((hitID << 24) | rayID);
                        }else{
                            state.prim_hit[hitID].valid = false;

                            if(!state.has_prim_hit()){
                                state.prim_batch_finished_count++;
                                traverse(rayID, trace_data->m_per_scalar_thread[tid]);
                            }
                        }
                    }
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

    uint32_t tlas_ptr, blas_ptr, bvh_ptr, tri_ptr, aabb_ptr;

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

void RTUnit::get_attr(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, std::vector<reg_data_t>& rd_data, uint32_t attr){
    impl_->get_attr(rs1_data, rs2_data, rd_data, attr);
}

void RTUnit::commit(const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, uint32_t action, RtuTraceData* trace_data){
    impl_->commit(rs1_data, rs2_data, action, trace_data);
}

void RTUnit::release_ray(const std::vector<reg_data_t>& rs1_data){
    impl_->release_ray(rs1_data);
}