#include "rt_core.h"
#include "rt_unit.h"

#define NODE_ADDR(root, idx) root + (idx) * sizeof(BVHNode)
#define INSTANCE_ADDR(root, idx) root + (idx) * sizeof(BLASNode)
#define TRI_ADDR(root, idx) root + (idx) * sizeof(Triangle)
#define PRIM_AABB_ADDR(root, idx) root + (idx) * sizeof(AABB)

using namespace vortex;

RTCore::RTCore(RTUnit* rt_unit, const DCRS &dcrs)
    : rt_unit_(rt_unit)
    , dcrs_(dcrs)
{}

void RTCore::allocate(uint32_t rayID, const Ray& world_ray, uint32_t tlas_addr, float tmin, float tmax){
    rays_[rayID] = world_ray;
    traversal_states_[rayID] = TraversalState(world_ray, tlas_addr, tmin, tmax);
}

void RTCore::traverse(TraversalState& state, per_thread_info &thread_info){
    while(state.status == TraversalStatus::TRACE){
        BVHNode node;
        dcache_read(&node, state.node_ptr, sizeof(BVHNode));
        thread_info.RT_mem_accesses.emplace_back(state.node_ptr, sizeof(BVHNode), TransactionType::BVH_INTERNAL_NODE);

        switch(node.type){
            case BVH_INTERNAL: {
                std::array<BoxHit, RT_BOX_INTERSECTION_WIDTH> box_hits;
                uint32_t valid_count = ray_nBox_intersect(node, state, box_hits); // SIMD intersection

                uint32_t k = state.trail[state.level];
                uint32_t start = (k == RT_BVH_WIDTH) ? valid_count - 1 : k;
                uint32_t end = valid_count;

                if(valid_count == 0 || start >= end){
                    state.pop();
                }else{
                    std::sort(box_hits.begin(), box_hits.end(), BoxHit::compare);

                    BoxHit closest = box_hits[start++];
                    state.node_ptr = NODE_ADDR(state.root_ptr, node.leftFirst + closest.idx);
                    
                    if(start == end){
                        state.trail[state.level] = RT_BVH_WIDTH;
                    }
                #if RT_STACK_SIZE > 0
                    else{
                        for (int32_t i = (int32_t)end - 1; i >= (int32_t)start; i--) {
                            uint32_t node_addr = NODE_ADDR(state.root_ptr, node.leftFirst + box_hits[i].idx);
                            bool isFarthest = (i == (int32_t)end - 1);
                            state.stack.push(node_addr | isFarthest); // encode one bit info into addr
                        }
                    }
                #endif
                    state.level++;
                }
                break;
            }
            case INSTANCE_LEAF: {
                state.instanceID = node.leftFirst;
                state.status = TraversalStatus::INSTANCE_HIT;
                break;
            }
            case TRIANGLE_LEAF: {
                state.prim_count = node.leaf.prim_count;
                state.prim_base_id = node.leftFirst;
                state.prim_batch_finished_count = 0;
                state.leaf_flags = node.leaf.flags;

                state.geometryIndex = node.leaf.geometryIndex;
                state.status = TraversalStatus::TRI_LEAF_HIT;
                break;
            }
            case PROCEDURAL_LEAF: {
                state.prim_count = node.leaf.prim_count;
                state.prim_base_id = node.leftFirst;
                state.prim_batch_finished_count = 0;
                state.leaf_flags = node.leaf.flags;

                state.geometryIndex = node.leaf.geometryIndex;
                state.status = TraversalStatus::PROCEDURAL_LEAF_HIT;
                break;
            }
            default: std::abort();
        }
    }
}

void RTCore::traverse(uint32_t rayID, per_thread_info &thread_info){
    uint32_t tlas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TLAS_PTR);
    uint32_t blas_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BLAS_PTR);
    uint32_t bvh_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_BVH_PTR);
    uint32_t tri_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_TRI_PTR);
    uint32_t aabb_ptr = dcrs_.base_dcrs.read(VX_DCR_BASE_RTX_AABB_PTR);

    TraversalState& state = traversal_states_[rayID];
    thread_info.rayID = rayID;
    while(1){
        switch(state.status){
            case TraversalStatus::TRACE: {
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
                state.node_ptr = NODE_ADDR(bvh_ptr, blas.bvh_offset);
                state.root_ptr = NODE_ADDR(bvh_ptr, blas.bvh_offset);
                state.root_level = state.level;
                state.status = TraversalStatus::TRACE;
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
                            shader_queue_push(ShaderType::ANYHIT, rayID);
                            thread_info.terminate = false;
                            return;
                        }

                        else if(state.leaf_flags == OPAQUE){
                            // Comparator Tree
                            // Assume RT_TRI_INTERSECTION_WIDTH == 4 for now
                            Hit h0 = Hit::min(state.prim_hit[0], state.prim_hit[1]);
                            Hit h1 = Hit::min(state.prim_hit[2], state.prim_hit[3]);
                            state.best_hit = Hit::min(h0, h1); 
                        }
                    }
                }

                state.pop();
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
                        shader_queue_push(ShaderType::INTERSECTION, rayID);
                        thread_info.terminate = false;
                        return;
                    }
                }

                state.pop();
                break;
            }

            case TraversalStatus::RESTART: {
                state.node_ptr = state.root_ptr;
                state.level = state.root_level;
                state.status = TraversalStatus::TRACE;
                break;
            }

            case TraversalStatus::FINISHED: {
                if(state.root_ptr == tlas_ptr || state.root_level == 0 || state.trail[0] == RT_BVH_WIDTH){
                    // TLAS Finished
                    shader_queue_push(state.best_hit.valid ? ShaderType::CLOSET : ShaderType::MISS, rayID);
                    thread_info.terminate = true;
                    return;
                }else{
                    // BLAS Finished (BLAS -> TLAS)
                    state.ray = rays_[rayID];
                    state.root_ptr = tlas_ptr;
                    state.root_level = 0;
                    state.pop();
                }
                break;
            }

            default: std::abort();
        }

    }
}

void RTCore::shader_queue_push(ShaderType type, uint32_t rayID){
    switch(type){
        case ShaderType::MISS:
        case ShaderType::CLOSET:
            shader_queues_[type].push(rayID);
            break;

        case ShaderType::ANYHIT:
        case ShaderType::INTERSECTION:
            for(uint32_t idx = 0; idx < RT_BOX_INTERSECTION_WIDTH; idx++){
                if(traversal_states_[rayID].prim_hit[idx].valid){
                    shader_queues_[type].push((idx << 28) | rayID);
                }
            }
            break;

        default: break;
    }
}

ShaderType RTCore::shader_queue_pop(uint32_t out_warp[SIMD_WIDTH], uint32_t& active_lanes){
    ShaderType targetType = ShaderType::MISS;
    if(shader_queues_[ShaderType::CLOSET].size() > shader_queues_[targetType].size()){
        targetType = ShaderType::CLOSET;
    }  

    if(shader_queues_[ShaderType::INTERSECTION].size() > shader_queues_[targetType].size()){
        targetType = ShaderType::INTERSECTION;
    }

    if(shader_queues_[ShaderType::ANYHIT].size() > shader_queues_[targetType].size()){
        targetType = ShaderType::ANYHIT;
    }

    active_lanes = shader_queues_[targetType].pop_warp(out_warp);
    return targetType;
}

void RTCore::commit(uint32_t rayID, uint32_t hitID, Hit hit, ShaderType type, per_thread_info &thread_info){
    TraversalState& state = traversal_states_[rayID];

    if(hit.valid && hit.t < state.best_hit.t){
        if(type == ShaderType::ANYHIT){
            state.best_hit = hit;
        }

        else if(type == ShaderType::INTERSECTION){
            if(state.leaf_flags == OPAQUE){
                state.best_hit = hit;
            }else{
                state.prim_hit[hitID] = hit;
                //return ShaderType::ANYHIT; // should return when !state.has_prim_hit()
            }
        }
    }
    
    state.prim_hit[hitID].valid = false;

    if(!state.has_prim_hit()){
        state.prim_batch_finished_count++;
        traverse(rayID, thread_info);
    }
}

Ray RTCore::ray_transform(const Ray &ray, float *T){
    float m00 = T[0], m01 = T[1], m02 = T[2], m03 = T[3];
    float m10 = T[4], m11 = T[5], m12 = T[6], m13 = T[7];
    float m20 = T[8], m21 = T[9], m22 = T[10], m23 = T[11];
    // float m30 = T[12], m31 = T[13], m32 = T[14], m33 = T[15];

    Ray T_ray;
    T_ray.ro_x = m00 * ray.ro_x + m01 * ray.ro_y + m02 * ray.ro_z + m03;
    T_ray.ro_y = m10 * ray.ro_x + m11 * ray.ro_y + m12 * ray.ro_z + m13;
    T_ray.ro_z = m20 * ray.ro_x + m21 * ray.ro_y + m22 * ray.ro_z + m23;

    T_ray.rd_x = m00 * ray.rd_x + m01 * ray.rd_y + m02 * ray.rd_z;
    T_ray.rd_y = m10 * ray.rd_x + m11 * ray.rd_y + m12 * ray.rd_z;
    T_ray.rd_z = m20 * ray.rd_x + m21 * ray.rd_y + m22 * ray.rd_z;
    return T_ray;
}

void RTCore::ray_box_intersect(const Ray &ray, float min_x, float min_y, float min_z, float max_x, float max_y, float max_z, float& t_near, float& t_far){
    float ro_x = ray.ro_x, ro_y = ray.ro_y, ro_z = ray.ro_z;
    float rd_x = ray.rd_x, rd_y = ray.rd_y, rd_z = ray.rd_z;
    float idir_x, idir_y, idir_z, tx1, tx2, ty1, ty2, tz1, tz2;

    idir_x = 1.0f / rd_x;
    idir_y = 1.0f / rd_y;
    idir_z = 1.0f / rd_z;
    tx1 = (min_x - ro_x) * idir_x;
    tx2 = (max_x - ro_x) * idir_x;
    t_near = std::min(tx1, tx2);
    t_far = std::max(tx1, tx2);
    ty1 = (min_y - ro_y) * idir_y;
    ty2 = (max_y - ro_y) * idir_y;
    t_near = std::max(t_near, std::min(ty1, ty2));
    t_far = std::min(t_far, std::max(ty1, ty2));
    tz1 = (min_z - ro_z) * idir_z;
    tz2 = (max_z - ro_z) * idir_z;
    t_near = std::max(t_near, std::min(tz1, tz2));
    t_far = std::min(t_far, std::max(tz1, tz2));
}

uint32_t RTCore::ray_nBox_intersect(BVHNode& node, TraversalState& state, std::array<BoxHit, RT_BOX_INTERSECTION_WIDTH>& box_hits){
    uint32_t valid_count = 0;
    for(uint32_t i=0; i<RT_BVH_WIDTH; i++){
        if(node.internal.children[i].meta == 0) continue;
        float min_x = node.internal.px + std::ldexp(float(node.internal.children[i].qaabb[0]), node.ex);
        float min_y = node.internal.py + std::ldexp(float(node.internal.children[i].qaabb[1]), node.ey);
        float min_z = node.internal.pz + std::ldexp(float(node.internal.children[i].qaabb[2]), node.ez);

        float max_x = node.internal.px + std::ldexp(float(node.internal.children[i].qaabb[3]), node.ex);
        float max_y = node.internal.py + std::ldexp(float(node.internal.children[i].qaabb[4]), node.ey);
        float max_z = node.internal.pz + std::ldexp(float(node.internal.children[i].qaabb[5]), node.ez);

        float t_near, t_far;
        ray_box_intersect(state.ray, min_x, min_y, min_z, max_x, max_y, max_z, t_near, t_far);

        if (t_near <= t_far && t_far > state.tmin && t_near < state.best_hit.t) {
            float t = std::max(t_near, state.tmin);
            box_hits[i] = BoxHit(t, i);
            valid_count++;
        }
    }
    return valid_count;
}

void RTCore::ray_nBox_intersect(AABB* aabbs, uint32_t primBaseID, uint32_t primCount, TraversalState& state){
    for(uint32_t i=0; i<primCount; i++){
        float min_x = aabbs[i].min[0];
        float min_y = aabbs[i].min[1];
        float min_z = aabbs[i].min[2];

        float max_x = aabbs[i].max[0];
        float max_y = aabbs[i].max[1];
        float max_z = aabbs[i].max[2];

        float t_near, t_far;
        ray_box_intersect(state.ray, min_x, min_y, min_z, max_x, max_y, max_z, t_near, t_far);

        if (t_near <= t_far && t_far > state.tmin && t_near < state.best_hit.t) {
            state.prim_hit[i].primitiveID = primBaseID + i;
            state.prim_hit[i].instanceID = state.instanceID;
            state.prim_hit[i].geometryIndex = state.geometryIndex;
            state.prim_hit[i].valid = true;
        }
    }
}

float RTCore::ray_tri_intersect(const Ray &ray, const Triangle &tri, float &u, float &v){
    float v0_x = tri.v0_x, v0_y = tri.v0_y, v0_z = tri.v0_z;
    float v1_x = tri.v1_x, v1_y = tri.v1_y, v1_z = tri.v1_z;
    float v2_x = tri.v2_x, v2_y = tri.v2_y, v2_z = tri.v2_z;

    float ro_x = ray.ro_x, ro_y = ray.ro_y, ro_z = ray.ro_z;
    float rd_x = ray.rd_x, rd_y = ray.rd_y, rd_z = ray.rd_z;
    
    float edge1_x = v1_x - v0_x;
    float edge1_y = v1_y - v0_y;
    float edge1_z = v1_z - v0_z;

    float edge2_x = v2_x - v0_x;
    float edge2_y = v2_y - v0_y;
    float edge2_z = v2_z - v0_z;

    float h_x = rd_y * edge2_z - rd_z * edge2_y;
    float h_y = rd_z * edge2_x - rd_x * edge2_z;
    float h_z = rd_x * edge2_y - rd_y * edge2_x;

    float a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z;
    if (fabs(a) < EPSILON){
        return LARGE_FLOAT;
    }

    float f = 1 / a;
    float s_x = ro_x - v0_x;
    float s_y = ro_y - v0_y;
    float s_z = ro_z - v0_z;

    float w1 = f * (s_x * h_x + s_y * h_y + s_z * h_z);
    if (w1 < 0 || w1 > 1){
        return LARGE_FLOAT;
    }
        
    float q_x = s_y * edge1_z - s_z * edge1_y;
    float q_y = s_z * edge1_x - s_x * edge1_z;
    float q_z = s_x * edge1_y - s_y * edge1_x;

    const float w2 = f * (rd_x * q_x + rd_y * q_y + rd_z * q_z);
    if (w2 < 0 || w1 + w2 > 1){
        return LARGE_FLOAT;
    }
        
    const float tf = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z);
    if (tf <= EPSILON){
        return LARGE_FLOAT;
    }

    u = w1;
    v = w2;
    return tf;
}

void RTCore::ray_nTri_intersect(Triangle* tris, uint32_t triBaseID, uint32_t triCount, TraversalState& state){
    for(uint32_t i=0; i<triCount; i++){        
        float u, v;
        float t = ray_tri_intersect(state.ray, tris[i], u, v);

        if (t < state.best_hit.t && t > state.tmin) {
            state.prim_hit[i] = Hit(t, u, v, triBaseID + i, state.instanceID);
            state.prim_hit[i].geometryIndex = state.geometryIndex;
        }
    }
}

void RTCore::dcache_read(void* data, uint64_t addr, uint32_t size) {
    rt_unit_->dcache_read(data, addr, size);
}