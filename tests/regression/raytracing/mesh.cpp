#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "mesh.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <unordered_map>

Mesh::Mesh(const std::filesystem::path& objFile, uint32_t geometry_idx, bool opaque)
: geometry_idx(geometry_idx)
, opaque_(opaque)
, transform(mat4_t::Identity())
{
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = objFile.parent_path().string();
    reader_config.triangulate = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(objFile.string(), reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader Error: " << reader.Error() << "\n";
        }
        std::abort();
    }

    if (!reader.Warning().empty()) {
        std::string warning = reader.Warning();

        if (warning.find("Both `d` and `Tr` parameters defined") == std::string::npos) {
            std::cout << "TinyObjReader Warning: " << warning << "\n";
        }
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& tiny_mats = reader.GetMaterials();

    std::unordered_map<std::string, int> loaded_textures;

    materials_.reserve(tiny_mats.size());

    for (const auto& mat : tiny_mats) {
        material_info_t m{};
        m.ambient  = {mat.ambient[0], mat.ambient[1], mat.ambient[2]};
        m.diffuse  = {mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]};
        m.specular = {mat.specular[0], mat.specular[1], mat.specular[2]};
        m.emissive = {mat.emission[0], mat.emission[1], mat.emission[2]};
        m.shininess = mat.shininess;
        m.ior       = mat.ior;
        m.dissolve  = mat.dissolve;

        std::filesystem::path texPath = objFile.parent_path() / mat.diffuse_texname;
        if (!mat.diffuse_texname.empty() && std::filesystem::exists(texPath)) {
            if (loaded_textures.find(mat.diffuse_texname) == loaded_textures.end()) {
                textures_.push_back(std::make_unique<Surface>(texPath.string().c_str()));
                loaded_textures[mat.diffuse_texname] = static_cast<int>(textures_.size() - 1);
            }
            m.diffuse_tex_id = loaded_textures[mat.diffuse_texname];
            m.tex_width  = textures_[m.diffuse_tex_id]->width();
            m.tex_height = textures_[m.diffuse_tex_id]->height();
            m.has_alpha  = textures_[m.diffuse_tex_id]->hasAlpha() ? 1 : 0;
        } else {
            m.diffuse_tex_id = -1;
            m.tex_width  = 0;
            m.tex_height = 0;
            m.has_alpha  = 0;
        }
        materials_.push_back(m);
    }

    if (materials_.empty()) {
        materials_.push_back(material_info_t{});
    }

    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shape.mesh.num_face_vertices[f]);
            assert(fv == 3 && "Mesh must be triangulated!");

            tri_t current_tri;
            tri_ex_t current_tri_ex;

            // Fetch the 3 indices for this triangle
            tinyobj::index_t idx0 = shape.mesh.indices[index_offset + 0];
            tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];

            // Extract Positions
            auto get_pos = [&](tinyobj::index_t idx) -> float3_t {
                return {
                    attrib.vertices[3 * size_t(idx.vertex_index) + 0],
                    attrib.vertices[3 * size_t(idx.vertex_index) + 1],
                    attrib.vertices[3 * size_t(idx.vertex_index) + 2]
                };
            };
            current_tri.v0 = get_pos(idx0);
            current_tri.v1 = get_pos(idx1);
            current_tri.v2 = get_pos(idx2);

            // Extract UVs (fallback to 0,0 if missing)
            auto get_uv = [&](tinyobj::index_t idx) -> float2_t {
                if (idx.texcoord_index >= 0) {
                    return {
                        attrib.texcoords[2 * size_t(idx.texcoord_index) + 0],
                        attrib.texcoords[2 * size_t(idx.texcoord_index) + 1]
                    };
                }
                return {0.0f, 0.0f};
            };
            current_tri_ex.uv0 = get_uv(idx0);
            current_tri_ex.uv1 = get_uv(idx1);
            current_tri_ex.uv2 = get_uv(idx2);

            // Extract Normals (fallback to geometric normal if missing)
            if (idx0.normal_index >= 0 && idx1.normal_index >= 0 && idx2.normal_index >= 0) {
                auto get_norm = [&](tinyobj::index_t idx) -> float3_t {
                    return {
                        attrib.normals[3 * size_t(idx.normal_index) + 0],
                        attrib.normals[3 * size_t(idx.normal_index) + 1],
                        attrib.normals[3 * size_t(idx.normal_index) + 2]
                    };
                };
                current_tri_ex.N0 = get_norm(idx0);
                current_tri_ex.N1 = get_norm(idx1);
                current_tri_ex.N2 = get_norm(idx2);
            } else {
                // Compute flat geometric normal
                float3_t edge1 = current_tri.v1 - current_tri.v0;
                float3_t edge2 = current_tri.v2 - current_tri.v0;
                float3_t geom_normal = normalize(cross(edge1, edge2)); 
                
                current_tri_ex.N0 = geom_normal;
                current_tri_ex.N1 = geom_normal;
                current_tri_ex.N2 = geom_normal;
            }

            // Map Material ID safely
            int mat_id = shape.mesh.material_ids[f];
            current_tri_ex.texId = (mat_id >= 0 && mat_id < materials_.size()) ? mat_id : 0;

            tri_.push_back(current_tri);
            triEx_.push_back(current_tri_ex);
            
            index_offset += 3;
        }
    }
}

Mesh::Mesh(tri_t tri, tri_ex_t triEx, material_info_t mat, uint32_t geometry_idx)
    : geometry_idx(geometry_idx), opaque_(true), procedural_(true) {
    tri_.push_back(tri);
    triEx_.push_back(triEx);
    materials_.push_back(mat);
}

Mesh* Mesh::CreateSphere(float3_t center, float radius, float3_t color, uint32_t geometry_idx) {
    // Fake tri_t so the BVH builder gets exact AABB bounds and centroid:
    //   v0 = center - r  → AABB min
    //   v1 = center      → centroid: (v0+v1+v2)/3 = center
    //   v2 = center + r  → AABB max
    tri_t tri;
    tri.v0 = center - float3_t(radius);
    tri.v1 = center;
    tri.v2 = center + float3_t(radius);

    tri_ex_t triEx{};
    triEx.texId = 0;

    material_info_t mat{};
    mat.diffuse        = color;
    mat.diffuse_tex_id = -1;

    auto* mesh = new Mesh(tri, triEx, mat, geometry_idx);

    shape_t shape{};
    shape.center = center;
    shape.radius = radius;
    //shape.type   = static_cast<uint32_t>(ShapeType::Sphere);
    mesh->shapes_.push_back(shape);

    AABB aabb{};
    aabb.bmin = center - float3_t(radius);
    aabb.bmax = center + float3_t(radius);
    mesh->aabbs_.push_back(aabb);

    return mesh;
}

std::vector<Surface*> Mesh::textures() const {
    std::vector<Surface*> raw_ptrs;
    raw_ptrs.reserve(textures_.size());
    for (const auto& tex : textures_) {
        raw_ptrs.push_back(tex.get());
    }
    return raw_ptrs;
}