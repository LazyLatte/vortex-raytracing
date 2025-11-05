#include "rt_unit.h"
#include "rt_ray_buffer.h"
#include "rt_op_unit.h"
#include "core.h"
#include <VX_config.h>
#include <vector>
#include <array>
#include <iostream>
#include <algorithm>
#include <cmath>
#define BVH_WIDTH 4
#define BVH_STACK_SIZE 64

#define TLAS_NODE_SIZE 32
#define BLAS_NODE_SIZE 160
#define BVH_NODE_SIZE 36
#define QBVH_NODE_SIZE 56
#define TRI_SIZE 36

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

    ~Impl() {
        // Destructor logic if needed
    }

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

            auto trace_data = std::dynamic_pointer_cast<RTUnit::MemTraceData>(trace->data);

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
        }

        //Question: pipline latency?
        //simobject_->Outputs.at(iw).push(trace, pipeline_latency);
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

    void read_vec3(uint32_t ptr, float &v0, float &v1, float &v2){
        uint32_t tmp;
        dcache_read(&tmp, ptr + 0 * sizeof(float), sizeof(float));
        v0 = *reinterpret_cast<float*>(&tmp);
        dcache_read(&tmp, ptr + 1 * sizeof(float), sizeof(float));
        v1 = *reinterpret_cast<float*>(&tmp);
        dcache_read(&tmp, ptr + 2 * sizeof(float), sizeof(float));
        v2 = *reinterpret_cast<float*>(&tmp);
    }

    void read_mat4(uint32_t ptr, float *m){
        dcache_read(m, ptr, sizeof(float) * 16);
    }

    float traverse(uint32_t wid, uint32_t tid, MemTraceData* trace_data){
        //std::cout << wid << " " << tid << std::endl;
        trace_data->pipeline_latency = 0;
        RayBuffer &ray_buffer = ray_buffers_.at(wid).at(tid);
        ray_buffer.hit.dist = LARGE_FLOAT;
        //traverse_tlas(0, ray_buffer, trace_data);

        uint32_t tlas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TLAS_PTR);
        uint32_t blas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BLAS_PTR);
        uint32_t bvh_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BVH_PTR);
        uint32_t qBvh_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_QBVH_PTR);
        uint32_t tri_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_PTR);
        uint32_t tri_idx_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_IDX_PTR);

        RestartTrailTraversal rtt(tlas_ptr, blas_ptr, bvh_ptr, qBvh_ptr, tri_ptr, tri_idx_ptr, this);

        rtt.traverse(ray_buffer, trace_data);

        return ray_buffer.hit.dist;
        //separate hit and ray???????????????
    }



    // void traverse_tlas(uint32_t tlas_root, RayBuffer &ray_buf, MemTraceData* trace_data){
    //     uint32_t tlas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TLAS_PTR);

    //     uint32_t stack[BVH_STACK_SIZE];
    //     uint32_t stackPtr = 0;
    //     stack[stackPtr++] = tlas_root;
        
    //     while (stackPtr != 0) {
    //         uint32_t nodeIdx = stack[--stackPtr];
    //         uint32_t tlas_node_ptr = tlas_ptr + nodeIdx * TLAS_NODE_SIZE;
            
    //         uint32_t leftRight;
    //         dcache_read(&leftRight, tlas_node_ptr + 3 * sizeof(float), sizeof(uint32_t));
    //         if (leftRight == 0) {
    //             uint32_t blasIdx;
    //             dcache_read(&blasIdx, tlas_node_ptr + 6 * sizeof(float) + sizeof(uint32_t), sizeof(uint32_t));
    //             traverse_blas(blasIdx, ray_buf, trace_data);
    //         } else {
                
    //             uint32_t left = leftRight & 0xFFFF;
    //             uint32_t right = leftRight >> 16;

    //             uint32_t left_node_ptr = tlas_ptr + left * TLAS_NODE_SIZE;
    //             uint32_t right_node_ptr = tlas_ptr + right * TLAS_NODE_SIZE;

    //             float min_x, min_y, min_z, max_x, max_y, max_z;
                
    //             read_vec3(left_node_ptr + 0 * sizeof(float), min_x, min_y, min_z);
    //             read_vec3(left_node_ptr + 4 * sizeof(float), max_x, max_y, max_z);

    //             float dLeft = ray_box_intersect(ray_buf.ray, min_x, min_y, min_z, max_x, max_y, max_z, trace_data->pipeline_latency);
                
    //             read_vec3(right_node_ptr + 0 * sizeof(float), min_x, min_y, min_z);
    //             read_vec3(right_node_ptr + 4 * sizeof(float), max_x, max_y, max_z);

    //             float dRight = ray_box_intersect(ray_buf.ray, min_x, min_y, min_z, max_x, max_y, max_z, trace_data->pipeline_latency);

    //             bool hitLeft = (dLeft != LARGE_FLOAT) && (dLeft < ray_buf.hit.dist);
    //             bool hitRight = (dRight != LARGE_FLOAT) && (dRight < ray_buf.hit.dist);

    //             if (hitLeft && hitRight) {
    //                 if (dLeft > dRight) {
    //                     std::swap(left, right);
    //                 }
    //                 stack[stackPtr++] = right;
    //                 stack[stackPtr++] = left;
    //                 } else if (hitLeft) {
    //                 stack[stackPtr++] = left;
    //                 } else if (hitRight) {
    //                 stack[stackPtr++] = right;
    //             }
                
    //         }
    //     }
    // }

    // void traverse_blas(uint32_t blasIdx, RayBuffer &ray_buf, MemTraceData* trace_data){
    //     uint32_t blas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BLAS_PTR);
    //     uint32_t blas_node_ptr = blas_ptr + blasIdx * BLAS_NODE_SIZE;
    //     // RayBuffer backup = ray; //problem: ray, hit should be separated
    
    //     // float inv00, inv01, inv02, inv03, 
    //     //     inv10, inv11, inv12, inv13, 
    //     //     inv20, inv21, inv22, inv23, 
    //     //     inv30, inv31, inv32, inv33;
        
    //     // read_mat4(
    //     //     blas_node_ptr + 16 * sizeof(float),
    //     //     inv00, inv01, inv02, inv03,
    //     //     inv10, inv11, inv12, inv13, 
    //     //     inv20, inv21, inv22, inv23, 
    //     //     inv30, inv31, inv32, inv33
    //     // );
        
    //     // backup.ro_x = inv00 * ray.ro_x + inv01 * ray.ro_y + inv02 * ray.ro_z + inv03;
    //     // backup.ro_y = inv10 * ray.ro_x + inv11 * ray.ro_y + inv12 * ray.ro_z + inv13;
    //     // backup.ro_z = inv20 * ray.ro_x + inv21 * ray.ro_y + inv22 * ray.ro_z + inv23;

    //     // backup.rd_x = inv00 * ray.rd_x + inv01 * ray.rd_y + inv02 * ray.rd_z;
    //     // backup.rd_y = inv10 * ray.rd_x + inv11 * ray.rd_y + inv12 * ray.rd_z;
    //     // backup.rd_z = inv20 * ray.rd_x + inv21 * ray.rd_y + inv22 * ray.rd_z;

    //     //pipeline_latency += TRANSFORM_LATENCY;

    //     uint32_t bvh_offset;
    //     dcache_read(&bvh_offset, blas_node_ptr + 32 * sizeof(float), sizeof(uint32_t));
        
    //     //traverse_bvh(bvh_offset, blasIdx, ray_buf, trace_data); //should use backup

    //     uint32_t bvh_base_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BVH_PTR);
    //     uint32_t tri_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_PTR);
    //     uint32_t tri_idx_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_IDX_PTR);
    //     uint32_t bvh_ptr = bvh_base_ptr + bvh_offset * BVH_NODE_SIZE;
    //     RestartTrailTraversal rtt(bvh_ptr, tri_ptr, tri_idx_ptr, blasIdx, this);
    //     //std::cout << "A" <<std::endl;
    //     rtt.traverse(ray_buf, trace_data);
    // }

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

    class RestartTrailTraversal{
        public:
            RestartTrailTraversal(
                uint32_t tlas_ptr,
                uint32_t blas_ptr,
                uint32_t bvh_ptr, 
                uint32_t qBvh_ptr,
                uint32_t tri_ptr, 
                uint32_t tri_idx_ptr,
                RTUnit::Impl* rt_unit
            );

            void traverse(RayBuffer &ray_buf, RTUnit::MemTraceData* trace_data);

        private:
            void push(uint32_t node_ptr, bool isLast);
            uint32_t pop(bool* terminate);
            int32_t findNextParentLevel();
            
            void dcache_read(void* data, uint64_t addr, uint32_t size);
            void read_vec3(uint32_t ptr, float &v0, float &v1, float &v2);
            bool read_node(uint32_t node_ptr, BVHNode *node);
            bool isTopLevel(BVHNode *node);
            bool isLeaf(BVHNode *node);
            uint32_t tlas_ptr, blas_ptr, bvh_ptr, qBvh_ptr, tri_ptr, tri_idx_ptr;

            std::array<uint32_t, 32> trail;
            uint32_t level;
            //uint32_t popLevel;
            
            ShortStack<StackEntry> traversal_stack;

            RTUnit::Impl* rt_unit_;
    };
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

float RTUnit::traverse(uint32_t wid, uint32_t tid, MemTraceData* trace_data){
    return impl_->traverse(wid, tid, trace_data);
}


RTUnit::Impl::RestartTrailTraversal::RestartTrailTraversal(
    uint32_t tlas_ptr,
    uint32_t blas_ptr,
    uint32_t bvh_ptr, 
    uint32_t qBvh_ptr,
    uint32_t tri_ptr, 
    uint32_t tri_idx_ptr, 
    RTUnit::Impl* rt_unit
): 
    tlas_ptr(tlas_ptr),
    blas_ptr(blas_ptr),
    bvh_ptr(bvh_ptr), 
    qBvh_ptr(qBvh_ptr),
    tri_ptr(tri_ptr), 
    tri_idx_ptr(tri_idx_ptr),
    traversal_stack(3),
    rt_unit_(rt_unit)
{}

bool cmp (HH ha, HH hb) { return (ha.dist > hb.dist); }
void RTUnit::Impl::RestartTrailTraversal::traverse(RayBuffer &ray_buf, RTUnit::MemTraceData* trace_data){
    level = 0;
    trail.fill(0);

    uint32_t blasIdx = 0;
    uint32_t base_ptr = tlas_ptr;
    uint32_t node_ptr = tlas_ptr;
    bool terminate = false;

    BVHNode node;
    float M[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float I[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    while(1){
        //rya transform
        while(read_node(node_ptr, &node)){            
            //bool topLevel = isTopLevel(&node);
            std::vector<HH> hits;

            for(int i=0; i<BVH_WIDTH; i++){
                float min_x = node.px + std::ldexp(float(node.children[i].qaabb[0]), node.ex);
                float min_y = node.py + std::ldexp(float(node.children[i].qaabb[1]), node.ey);
                float min_z = node.pz + std::ldexp(float(node.children[i].qaabb[2]), node.ez);

                float max_x = node.px + std::ldexp(float(node.children[i].qaabb[3]), node.ex);
                float max_y = node.py + std::ldexp(float(node.children[i].qaabb[4]), node.ey);
                float max_z = node.pz + std::ldexp(float(node.children[i].qaabb[5]), node.ez);

                float d = ray_box_intersect(ray_buf.ray, min_x, min_y, min_z, max_x, max_y, max_z, /*topLevel ? I :*/ M, trace_data->pipeline_latency);

                if(d < ray_buf.hit.dist){
                    hits.emplace_back(d, i);
                }
            }

            std::sort(hits.begin(), hits.end(), cmp);

            uint32_t k = trail[level];
            uint32_t dropCount = (k == BVH_WIDTH) ? hits.size() - 1 : k;
            for(int i=0; i<dropCount; i++){
                if(hits.size() > 0){
                    hits.pop_back();
                }
            }
            
            if(hits.size() == 0){
                node_ptr = pop(&terminate);
                if(terminate) return;
            }else{
                HH h = hits.back();
                hits.pop_back();

                uint32_t nodeIdx = node.leftRight + h.childIdx;
                node_ptr = base_ptr + nodeIdx * sizeof(BVHNode);
                
                if(hits.size() == 0){
                    trail[level] = BVH_WIDTH;
                }else{
                    //mark
                    for(auto iter = hits.begin(); iter != hits.end(); iter++){
                        nodeIdx = node.leftRight + (*iter).childIdx;
                        push(base_ptr + nodeIdx * sizeof(BVHNode), iter == hits.begin());
                    }
                }
                level++;
            }

        }

        if(isTopLevel(&node)){
            blasIdx = node.leafIdx;
            uint32_t bvh_offset;
            uint32_t blas_node_ptr = blas_ptr + blasIdx * BLAS_NODE_SIZE;
            dcache_read(&bvh_offset, blas_node_ptr + 32 * sizeof(float), sizeof(uint32_t));
            dcache_read(&M[0], blas_node_ptr + 16 * sizeof(float), 16 * sizeof(float));
            node_ptr = qBvh_ptr + bvh_offset * sizeof(BVHNode);
            base_ptr = qBvh_ptr;
        }else{
            
            uint32_t triCount = node.leafIdx;
            uint32_t leftFirst = node.leftRight;
            for (uint32_t i = 0; i < triCount; ++i) {
                uint32_t triIdx;
                dcache_read(&triIdx, tri_idx_ptr + (leftFirst + i) * sizeof(uint32_t), sizeof(uint32_t));
                
                uint32_t tri_addr = tri_ptr + triIdx * TRI_SIZE;

                float v0_x, v0_y, v0_z, v1_x, v1_y, v1_z, v2_x, v2_y, v2_z;
                read_vec3(tri_addr + 0 , v0_x, v0_y, v0_z);
                read_vec3(tri_addr + 12, v1_x, v1_y, v1_z);
                read_vec3(tri_addr + 24, v2_x, v2_y, v2_z);

                float bx, by, bz;
                float d = ray_tri_intersect(
                    ray_buf.ray,
                    v0_x, v0_y, v0_z, 
                    v1_x, v1_y, v1_z, 
                    v2_x, v2_y, v2_z, 
                    bx, by, bz, M,
                    trace_data->pipeline_latency
                );
                
                if (d != LARGE_FLOAT && d < ray_buf.hit.dist) {
                    ray_buf.hit.dist = d;
                    ray_buf.hit.bx = bx;
                    ray_buf.hit.by = by;
                    ray_buf.hit.bz = bz;
                    ray_buf.hit.blasIdx = blasIdx;
                    ray_buf.hit.triIdx = triIdx;
                }
            }

            node_ptr = pop(&terminate);
            if(terminate) return;    
        }
    }
}


void RTUnit::Impl::RestartTrailTraversal::push(uint32_t node_ptr, bool isLast){
    StackEntry se;
    se.node_ptr = node_ptr;
    se.isLast = isLast;
    traversal_stack.push(se);
}

int32_t RTUnit::Impl::RestartTrailTraversal::findNextParentLevel(){
    for(int i=level-1; i>=0; i--){
        if(trail[i] != BVH_WIDTH){
            return i;
        }
    }
    return -1;
}

uint32_t RTUnit::Impl::RestartTrailTraversal::pop(bool* terminate){
    int32_t parentLevel = findNextParentLevel();

    if(parentLevel < 0){
        *terminate = true;
        return -1;
    }

    trail[parentLevel]++;

    for(int i=parentLevel+1; i<32; i++){
        trail[i] = 0;
    }

    uint32_t node_ptr;
    if(traversal_stack.empty()){
        node_ptr = tlas_ptr;
        //base_ptr...
        level = 0;
        //std::cout << "Restart..." << std::endl;
    }else{
        StackEntry se = traversal_stack.pop();
        node_ptr = se.node_ptr;
        if(se.isLast){
            trail[parentLevel] = BVH_WIDTH;
        }

        level = parentLevel + 1;
    }

    return node_ptr;
}

bool RTUnit::Impl::RestartTrailTraversal::read_node(uint32_t node_ptr, BVHNode *node){
    dcache_read(node, node_ptr, sizeof(BVHNode));
    return !isLeaf(node);
}

bool RTUnit::Impl::RestartTrailTraversal::isTopLevel(BVHNode *node){
    return (uint32_t)(node->imask) == 1;
}

bool RTUnit::Impl::RestartTrailTraversal::isLeaf(BVHNode *node){
    return (isTopLevel(node) && node->leftRight == 0) || (!isTopLevel(node) && node->leafIdx != 0);
}

void RTUnit::Impl::RestartTrailTraversal::dcache_read(void* data, uint64_t addr, uint32_t size) {
    rt_unit_->dcache_read(data, addr, size);
}

void RTUnit::Impl::RestartTrailTraversal::read_vec3(uint32_t ptr, float &v0, float &v1, float &v2){
    rt_unit_->read_vec3(ptr, v0, v1, v2);
}