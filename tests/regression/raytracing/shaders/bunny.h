#pragma once
#include "shader.h"
#include <vx_raytrace.h>

namespace Shader {
namespace Bunny {

inline void CHS(ray_payload_t *payload, kernel_arg_t *arg) {
    uint32_t _t = vortex::rt::get_attr<VX_RT_HIT_T>();
    uint32_t _u = vortex::rt::get_attr<VX_RT_HIT_ATTR_U>();
    uint32_t _v = vortex::rt::get_attr<VX_RT_HIT_ATTR_V>();
    uint32_t instanceID  = vortex::rt::get_attr<VX_RT_HIT_INSTANCE_ID>();
    uint32_t primitiveID = vortex::rt::get_attr<VX_RT_HIT_PRIMITIVE_ID>();

    float t  = *reinterpret_cast<float*>(&_t);
    float bx = *reinterpret_cast<float*>(&_u);
    float by = *reinterpret_cast<float*>(&_v);
    float bz = 1.0f - bx - by;

    auto triEx_ptr = reinterpret_cast<const tri_ex_t *>(arg->triEx_addr);
    const tri_ex_t &triEx = triEx_ptr[primitiveID];

    auto blas_ptr = reinterpret_cast<const blas_node_t *>(arg->blas_addr);
    const blas_node_t &blas = blas_ptr[instanceID];

    float3_t I = payload->origin + payload->direction * t;

    float3_t N = triEx.N1 * bx + triEx.N2 * by + triEx.N0 * bz;
    mat4_t invTranspose = blas.invTransform.transposed();
    N = normalize(TransformVector(N, invTranspose));

    auto mat_ptr = reinterpret_cast<const material_info_t *>(arg->mat_addr);
    const material_info_t &mat = mat_ptr[triEx.texId];

    // Diffuse albedo
    float3_t albedo = mat.diffuse;

    {
        float3_t L = normalize(arg->light_pos - I);
        float NdotL = std::max(dot(N, L), 0.0f);
        payload->irradiance += payload->throughput * albedo * (NdotL + 0.2f);
    }

    payload->stop = true;
}

}
}

