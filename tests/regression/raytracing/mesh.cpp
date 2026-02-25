#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "mesh.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <unordered_map>

// Mesh class implementation

// std::vector<std::string> split(const std::string& str, char delimiter) {
//   std::vector<std::string> tokens;
//   std::string token;
//   std::istringstream tokenStream(str);
//   while (std::getline(tokenStream, token, delimiter)) {
//     tokens.push_back(token);
//   }
//   return tokens;
// }

// int load_obj(const char *objFile, obj_mesh_t &mesh) {
//   std::ifstream file(objFile);
//   if (!file.is_open()) {
//     fprintf(stderr, "Error: Could not open file %s\n", objFile);
// 		return -1;
// 	}

//   mesh.positions.clear();
//   mesh.normals.clear();
//   mesh.texcoords.clear();
//   mesh.vertices.clear();
//   mesh.faces.clear();

//   std::unordered_map<std::string, uint32_t> vertex_cache;

//   std::string line;
//   while (std::getline(file, line)) {
//     std::istringstream iss(line);
//     std::string type;
//     iss >> type;

//     if (type == "v") { // Vertex position
//       float3_t v;
//       iss >> v.x >> v.y >> v.z;
//       mesh.positions.push_back(v);
//     } else if (type == "vn") { // Vertex normal
//       float3_t n;
//       iss >> n.x >> n.y >> n.z;
//       mesh.normals.push_back(n);
//     } else if (type == "vt") { // Texture coordinate (stored as float3 for alignment)
//       float2_t tc;
//       iss >> tc.x >> tc.y;
//       mesh.texcoords.push_back(tc);
//     } else if (type == "f") { // Face (triangulated)
//       std::vector<std::string> verts;
//       std::string vertex;
//       while (iss >> vertex)
//         verts.push_back(vertex);

//       // We build triangles (0, j, j+1) for j=1…verts.size()-2
//       for (size_t j = 1; j + 1 < verts.size(); ++j) {
//         obj_mesh_t::face_t face;
//         std::array<size_t,3> idx = { 0, j, j+1 };
//         for (int k = 0; k < 3; ++k) {
//           auto parts = split(verts[idx[k]], '/');
//           uint32_t p = std::stoi(parts[0]) - 1;
//           uint32_t t = (parts.size()>1 && !parts[1].empty()) ? std::stoi(parts[1]) - 1 : 0;
//           uint32_t n = (parts.size()>2 && !parts[2].empty()) ? std::stoi(parts[2]) - 1 : 0;
//           std::string key = std::to_string(p)+"/"+std::to_string(t)+"/"+std::to_string(n);
//           auto it = vertex_cache.find(key);
//           if (it != vertex_cache.end()) {
//             face.v[k] = it->second;
//           } else {
//             mesh.vertices.push_back({p,n,t});
//             face.v[k] = mesh.vertices.size() - 1;
//             vertex_cache[key] = face.v[k];
//           }
//         }
//         mesh.faces.push_back(face);
//       }
//     }
//   }

//   return 0; // Success
// }

// Mesh::Mesh(const char *objFile, const char *texFile, float reflectivity)
//   : reflectivity_(reflectivity) {
//   obj_mesh_t obj;
// 	if (load_obj(objFile, obj) != 0) {
// 		std::abort();
// 	}

// 	auto triCount = obj.faces.size();
// 	tri_.resize(triCount);
// 	triEx_.resize(triCount);

// 	for (uint32_t i = 0; i < triCount; i++) {
// 		const obj_mesh_t::face_t &face = obj.faces[i];
// 		const obj_mesh_t::vert_t &v0 = obj.vertices[face.v[0]];
// 		const obj_mesh_t::vert_t &v1 = obj.vertices[face.v[1]];
// 		const obj_mesh_t::vert_t &v2 = obj.vertices[face.v[2]];
// 		tri_[i].v0 = obj.positions[v0.p];
// 		tri_[i].v1 = obj.positions[v1.p];
// 		tri_[i].v2 = obj.positions[v2.p];

// 		triEx_[i].N0 = obj.normals[v0.n];
// 		triEx_[i].N1 = obj.normals[v1.n];
// 		triEx_[i].N2 = obj.normals[v2.n];

// 		triEx_[i].uv0 = obj.texcoords[v0.t];
// 		triEx_[i].uv1 = obj.texcoords[v1.t];
// 		triEx_[i].uv2 = obj.texcoords[v2.t];
// 	}

// 	// load texture
// 	texture_ = new Surface(texFile);
// }

Mesh::~Mesh() {
  for (auto tex : textures_) {
      delete tex;
  }
  textures_.clear();
}


int load_obj_tiny(const char *objFile, obj_mesh_t &mesh, std::vector<material_info_t>& materials_out, std::vector<material_textures_t>& textures_out) {
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = std::filesystem::path(objFile).parent_path().string(); 

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(objFile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        return -1;
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    // 1. Process Materials
    for (const auto& mat : materials) {
        material_info_t m;
        // Map colors
        m.ambient  = {mat.ambient[0], mat.ambient[1], mat.ambient[2]};
        m.diffuse  = {mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]};
        m.specular = {mat.specular[0], mat.specular[1], mat.specular[2]};
        m.emissive = {mat.emission[0], mat.emission[1], mat.emission[2]};

        // Map scalars
        m.shininess    = mat.shininess;
        m.ior          = mat.ior;
        m.dissolve     = mat.dissolve;

        material_textures_t tex_names;
        tex_names.ambient_texname = mat.ambient_texname; 
        tex_names.diffuse_texname = mat.diffuse_texname; 
        tex_names.specular_texname = mat.specular_texname; 
        textures_out.push_back(tex_names);

        materials_out.push_back(m);
    }

    // 2. Process Geometry
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            // Sponza is usually triangulated, but this ensures it
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            obj_mesh_t::face_t face;

            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                // Vertex Data
                float3_t pos = {
                    attrib.vertices[3 * size_t(idx.vertex_index) + 0],
                    attrib.vertices[3 * size_t(idx.vertex_index) + 1],
                    attrib.vertices[3 * size_t(idx.vertex_index) + 2]
                };
                
                // Normal Data (optional check if they exist)
                float3_t norm = {0,0,0};
                if (idx.normal_index >= 0) {
                    norm = {
                        attrib.normals[3 * size_t(idx.normal_index) + 0],
                        attrib.normals[3 * size_t(idx.normal_index) + 1],
                        attrib.normals[3 * size_t(idx.normal_index) + 2]
                    };
                }

                // UV Data
                float2_t uv = {0,0};
                if (idx.texcoord_index >= 0) {
                    uv = {
                        attrib.texcoords[2 * size_t(idx.texcoord_index) + 0],
                        attrib.texcoords[2 * size_t(idx.texcoord_index) + 1]
                    };
                }

                // In your existing structure, you cache vertices to handle indexing.
                // For simplicity here, we add them directly:
                mesh.positions.push_back(pos);
                mesh.normals.push_back(norm);
                mesh.texcoords.push_back(uv);
                
                uint32_t new_idx = mesh.positions.size() - 1;
                mesh.vertices.push_back({new_idx, new_idx, new_idx});
                face.v[v] = new_idx;
            }
            // Store material ID for this triangle (crucial for Sponza!)
            face.material_id = shapes[s].mesh.material_ids[f];
            mesh.faces.push_back(face);
            index_offset += fv;
        }
    }
    return 0;
}

Mesh::Mesh(const char *objFile){
  obj_mesh_t obj;
  std::vector<material_info_t> mats;
  std::vector<material_textures_t> textures;
  if (load_obj_tiny(objFile, obj, mats, textures) != 0) {
      std::cerr << "Failed to load " << objFile << std::endl;
      std::abort();
  }

  std::string baseDir = std::filesystem::path(objFile).parent_path().string() + "/";
  std::unordered_map<std::string, int> loaded_textures;

  assert(mats.size() == textures.size());

  for(int i=0; i<mats.size(); i++){
    auto& m = mats[i];
    auto& tex = textures[i];
    auto& diffuse_name = tex.diffuse_texname;
    if (!diffuse_name.empty()) {
      // Check if we already loaded this specific image
      if (loaded_textures.find(diffuse_name) == loaded_textures.end()) {
        std::string fullPath = baseDir + diffuse_name;
        Surface* newSurf = new Surface(fullPath.c_str());
        textures_.push_back(newSurf);
        loaded_textures[diffuse_name] = textures_.size() - 1;

        //std::cout << diffuse_name.c_str() << std::endl;
      }
      
      // Assign the shared ID and metadata
      m.diffuse_tex_id = loaded_textures[diffuse_name];
      m.tex_width  = textures_[m.diffuse_tex_id]->width();
      m.tex_height = textures_[m.diffuse_tex_id]->height();
    } else {
      m.diffuse_tex_id = -1;
      m.tex_width = 0;
      m.tex_height = 0;
    }
  }

  materials_ = mats;

  // 2. Populate Triangles
  size_t triCount = obj.faces.size();
  tri_.resize(triCount);
  triEx_.resize(triCount);

  for (uint32_t i = 0; i < triCount; i++) {
    const auto &f = obj.faces[i];
    
    const auto &v0 = obj.vertices[f.v[0]];
    const auto &v1 = obj.vertices[f.v[1]];
    const auto &v2 = obj.vertices[f.v[2]];

    tri_[i].v0 = obj.positions[v0.p];
    tri_[i].v1 = obj.positions[v1.p];
    tri_[i].v2 = obj.positions[v2.p];

    triEx_[i].N0 = obj.normals[v0.n];
    triEx_[i].N1 = obj.normals[v1.n];
    triEx_[i].N2 = obj.normals[v2.n];

    triEx_[i].uv0 = obj.texcoords[v0.t];
    triEx_[i].uv1 = obj.texcoords[v1.t];
    triEx_[i].uv2 = obj.texcoords[v2.t];

    // Safety: Ensure material_id is valid
    triEx_[i].texId = (f.material_id >= 0) ? f.material_id : 0; 
  }
}