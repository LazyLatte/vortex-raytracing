#pragma once
#include "rt_ray_buffer.h"
#include <algorithm>
#include <math.h> 
#define LARGE_FLOAT 1e30f
#define EPSILON 1e-6f

#define TRANSFORM_LATENCY 6
#define RAY_BOX_INTERSECTION_LATENCY 24
#define RAY_TRIANGLE_INTERSECTION_LATENCY 16

namespace vortex {
    Ray ray_transform(Ray ray, float *transform_matrix){     
        //latency += TRANSFORM_LATENCY;

        float m00 = transform_matrix[0];
        float m01 = transform_matrix[1];
        float m02 = transform_matrix[2];
        float m03 = transform_matrix[3];

        float m10 = transform_matrix[4];
        float m11 = transform_matrix[5];
        float m12 = transform_matrix[6];
        float m13 = transform_matrix[7];
        
        float m20 = transform_matrix[8];
        float m21 = transform_matrix[9];
        float m22 = transform_matrix[10];
        float m23 = transform_matrix[11];

        // float m30 = transform_matrix[12];
        // float m31 = transform_matrix[13];
        // float m32 = transform_matrix[14];
        // float m33 = transform_matrix[15];

        Ray transformed_ray;
        transformed_ray.ro_x = m00 * ray.ro_x + m01 * ray.ro_y + m02 * ray.ro_z + m03;
        transformed_ray.ro_y = m10 * ray.ro_x + m11 * ray.ro_y + m12 * ray.ro_z + m13;
        transformed_ray.ro_z = m20 * ray.ro_x + m21 * ray.ro_y + m22 * ray.ro_z + m23;

        transformed_ray.rd_x = m00 * ray.rd_x + m01 * ray.rd_y + m02 * ray.rd_z;
        transformed_ray.rd_y = m10 * ray.rd_x + m11 * ray.rd_y + m12 * ray.rd_z;
        transformed_ray.rd_z = m20 * ray.rd_x + m21 * ray.rd_y + m22 * ray.rd_z;
        return transformed_ray;
    }

    float ray_tri_intersect(
        Ray ray_,
        float v0_x, float v0_y, float v0_z,
        float v1_x, float v1_y, float v1_z,
        float v2_x, float v2_y, float v2_z,
        float &bx, float &by, float &bz,
        float *m, uint32_t& latency
    ){
        Ray ray = ray_transform(ray_, m);
        latency += RAY_TRIANGLE_INTERSECTION_LATENCY;

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

        bx = w1;
        by = w2;
        bz = 1 - w1 - w2;
        return tf;
    }

    float ray_box_intersect(
        Ray ray_,
        float min_x, float min_y, float min_z, 
        float max_x, float max_y, float max_z,
        float *m, uint32_t& latency
    ){
        Ray ray = ray_transform(ray_, m);
        latency += RAY_BOX_INTERSECTION_LATENCY;
        float ro_x = ray.ro_x, ro_y = ray.ro_y, ro_z = ray.ro_z;
        float rd_x = ray.rd_x, rd_y = ray.rd_y, rd_z = ray.rd_z;
        float idir_x, idir_y, idir_z, tmin, tmax, tx1, tx2, ty1, ty2, tz1, tz2;

        idir_x = 1.0f / rd_x;
        idir_y = 1.0f / rd_y;
        idir_z = 1.0f / rd_z;
        tx1 = (min_x - ro_x) * idir_x;
        tx2 = (max_x - ro_x) * idir_x;
        tmin = std::min(tx1, tx2);
        tmax = std::max(tx1, tx2);
        ty1 = (min_y - ro_y) * idir_y;
        ty2 = (max_y - ro_y) * idir_y;
        tmin = std::max(tmin, std::min(ty1, ty2));
        tmax = std::min(tmax, std::max(ty1, ty2));
        tz1 = (min_z - ro_z) * idir_z;
        tz2 = (max_z - ro_z) * idir_z;
        tmin = std::max(tmin, std::min(tz1, tz2));
        tmax = std::min(tmax, std::max(tz1, tz2));
        return tmax < tmin || tmax <= 0 ? LARGE_FLOAT : tmin;
    }


}