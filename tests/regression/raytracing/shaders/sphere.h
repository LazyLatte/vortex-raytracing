#pragma once
#include "shader.h"
#include <vx_raytrace.h>

namespace Shader {

namespace Sphere {

inline void IS(kernel_arg_t *arg) {
    uint32_t ox = vortex::rt::get_attr<VX_RT_OBJECT_RAY_RO_X>();
    uint32_t oy = vortex::rt::get_attr<VX_RT_OBJECT_RAY_RO_Y>();
    uint32_t oz = vortex::rt::get_attr<VX_RT_OBJECT_RAY_RO_Z>();
    uint32_t dx = vortex::rt::get_attr<VX_RT_OBJECT_RAY_RD_X>();
    uint32_t dy = vortex::rt::get_attr<VX_RT_OBJECT_RAY_RD_Y>();
    uint32_t dz = vortex::rt::get_attr<VX_RT_OBJECT_RAY_RD_Z>();
    uint32_t _tmin = vortex::rt::get_attr<VX_RT_T_MIN>();
    uint32_t _tmax = vortex::rt::get_attr<VX_RT_T_MAX>();
    uint32_t primitiveID = vortex::rt::get_attr<VX_RT_HIT_PRIMITIVE_ID>();

    float ro_x = *reinterpret_cast<float*>(&ox);
    float ro_y = *reinterpret_cast<float*>(&oy);
    float ro_z = *reinterpret_cast<float*>(&oz);
    float rd_x = *reinterpret_cast<float*>(&dx);
    float rd_y = *reinterpret_cast<float*>(&dy);
    float rd_z = *reinterpret_cast<float*>(&dz);
    float tmin = *reinterpret_cast<float*>(&_tmin);
    float tmax = *reinterpret_cast<float*>(&_tmax);

    auto shape_ptr = reinterpret_cast<const shape_t*>(arg->shape_addr);
    const shape_t& shape = shape_ptr[primitiveID];
    float cx = shape.center.x, cy = shape.center.y, cz = shape.center.z;
    float radius = shape.radius;

    // Solve |O + t·D − C|² = r²
    float ocx = ro_x - cx, ocy = ro_y - cy, ocz = ro_z - cz;
    float a    = rd_x*rd_x + rd_y*rd_y + rd_z*rd_z;
    float b    = 2.0f * (ocx*rd_x + ocy*rd_y + ocz*rd_z);
    float c    = ocx*ocx + ocy*ocy + ocz*ocz - radius*radius;
    float disc = b*b - 4.0f*a*c;

    if (disc < 0.0f) {
        vortex::rt::commit<VX_RT_INTERSECTION_IGNORE>();
        return;
    }

    float sqrt_disc = sqrtf(disc);
    float t = (-b - sqrt_disc) / (2.0f * a);   // near root first
    if (t < tmin || t > tmax) {
        t = (-b + sqrt_disc) / (2.0f * a);      // try far root
        if (t < tmin || t > tmax) {
            vortex::rt::commit<VX_RT_INTERSECTION_IGNORE>();
            return;
        }
    }

    vortex::rt::set_attr<VX_RT_HIT_T>(t);
    vortex::rt::commit<VX_RT_INTERSECTION_ACCEPT>();
}

inline void CHS(ray_payload_t *payload, kernel_arg_t *arg) {
    uint32_t _t = vortex::rt::get_attr<VX_RT_HIT_T>();
    uint32_t primitiveID = vortex::rt::get_attr<VX_RT_HIT_PRIMITIVE_ID>();

    float t = *reinterpret_cast<float*>(&_t);

    auto shape_ptr = reinterpret_cast<const shape_t*>(arg->shape_addr);
    auto triEx_ptr = reinterpret_cast<const tri_ex_t*>(arg->triEx_addr);
    auto mat_ptr   = reinterpret_cast<const material_info_t*>(arg->mat_addr);

    const shape_t& shape = shape_ptr[primitiveID];
    const material_info_t& mat = mat_ptr[triEx_ptr[primitiveID].texId];

    float3_t I = payload->origin + payload->direction * t;
    float3_t outward_N = normalize(I - shape.center);
    bool front_face = dot(payload->direction, outward_N) < 0.0f;
    float3_t N = front_face ? outward_N : -outward_N;

    switch (mat.illum) {
        case MaterialDielectric: {
            float3_t unit_dir = normalize(payload->direction);
            float etai_over_etat = front_face ? (1.0f / mat.ior) : mat.ior;
            float cos_theta = std::min(dot(-unit_dir, N), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

            float3_t new_dir;
            if (etai_over_etat * sin_theta > 1.0f) {
                new_dir = reflect(unit_dir, N);
            } else {
                new_dir = refract(unit_dir, N, etai_over_etat);
            }

            payload->origin    = I + new_dir * 1e-4f;
            payload->direction = new_dir;
            ++payload->bounce;
            break;
        }
        case MaterialMetallic: {
            payload->throughput *= mat.diffuse;
            payload->origin    = I + N * 1e-4f;
            payload->direction = reflect(normalize(payload->direction), N);
            ++payload->bounce;
            break;
        }
        default: {
            // Lambertian
            payload->irradiance += payload->throughput * diffuseLighting(
                I, N, mat.diffuse,
                arg->ambient_color, arg->light_color, arg->light_pos);
            payload->stop = true;
            break;
        }
    }
}

} // namespace Sphere

} // namespace Shader
