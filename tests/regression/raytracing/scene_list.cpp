#include "scene_list.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <random>
#include <functional>

static std::string resolve_path(const std::string& filename, const std::string& searchPaths) {
  std::ifstream ifs(filename);
  if (!ifs) {
    std::stringstream ss(searchPaths);
    std::string path;
    while (std::getline(ss, path, ',')) {
      if (!path.empty()) {
        std::string filePath = path + "/" + filename;
        std::ifstream ifs(filePath);
        if (ifs)
          return filePath;
      }
    }
  }
  return filename;
}

Scene* SceneList::CornellBox(){
    auto s_box = resolve_path(std::string("assets/") + "CornellBox/CornellBox-Original.obj", ASSETS_PATHS);
    auto* boxMesh = new Mesh(s_box, Geomrtry::CornellBox);

    Scene* scene = new Scene({boxMesh});
    scene->camera_pos = float3_t(0.0f, 1.0f, 3.6f);
    scene->camera_front = float3_t(0.0f, 0.0f, -1.0f);
    scene->camera_fov = 2.74747741945f;

    scene->light_pos = float3_t(-0.005f, 1.98f, -0.003f);
    scene->background_color = float3_t(0.0f, 0.0f, 0.0f);
    return scene;
}

Scene* SceneList::Bunny() {
    auto s_box   = resolve_path(std::string("assets/") + "CornellBox/CornellBox-Empty-RG.obj", ASSETS_PATHS);
    auto s_bunny = resolve_path("assets/bunny.obj", ASSETS_PATHS);

    auto* boxMesh   = new Mesh(s_box,   Geomrtry::CornellBox);
    auto* bunnyMesh = new Mesh(s_bunny, Geomrtry::Bunny);

    const mat4_t T = mat4_t::Translate(float3_t(-0.37837837837f, 0.01801801801f, -0.11711711711f)) 
        * mat4_t::Scale(0.72072072072f) 
        * mat4_t::RotateY(75.0f * PI / 180.0f);
        
    bunnyMesh->setTransform(T);
    bunnyMesh->setAllDiffuse(float3_t(0.8f, 0.8f, 0.8f));

    Scene* scene = new Scene({boxMesh, bunnyMesh});
    scene->camera_pos   = float3_t(0.0f, 1.0f, 3.6f);
    scene->camera_front = float3_t(0.0f, 0.0f, -1.0f);
    scene->camera_fov   = 2.74747741945f;

    scene->light_pos        = float3_t(-0.005f, 1.98f, -0.003f);
    scene->background_color = float3_t(0.0f, 0.0f, 0.0f);
    return scene;
}

Scene* SceneList::Sponza(){
    auto s_model = resolve_path(std::string("assets/") + "Sponza/sponza.obj", ASSETS_PATHS);
    auto* modelMesh = new Mesh(s_model, Geomrtry::Sponza);

    // create scene
    Scene* scene = new Scene({modelMesh});
    scene->camera_pos = float3_t(0.0f, 115.0f, 2.0f);
    scene->camera_front = float3_t(-1.0, 0.0, 0.0);
    scene->camera_fov = 1.428f;

    scene->light_pos = float3_t(0, 10, -10);
    scene->light_color = float3_t(1, 1, 1.0);
    scene->ambient_color = float3_t(0.4f, 0.4f, 0.4f);
    scene->background_color = float3_t(0.4f, 0.35f, 0.25f);
    return scene;
}

Scene* SceneList::Carnival(){
    auto s_model = resolve_path(std::string("assets/") + "Carnival/TheCarnival.obj", ASSETS_PATHS);
    auto* modelMesh = new Mesh(s_model, Geomrtry::Sponza);

    const mat4_t T = mat4_t::Translate(float3_t(400, 20, 150)) * mat4_t::Scale(0.2f);
    modelMesh->setTransform(T);

    // create scene
    Scene* scene = new Scene({modelMesh});
    scene->camera_pos = float3_t(275, 60, -700);
    scene->camera_front = normalize(float3_t(225, 120, -1000) -  scene->camera_pos);
    scene->camera_fov = 2.14450692051f;

    scene->light_pos = float3_t(0, 0, 0);
    scene->background_color = float3_t(0.5f, 0.7f, 1.0f);
    return scene;
}

Scene* SceneList::RayTracingInOneWeekend() {
    struct SphereInfo {
        float3_t center;
        float    radius;
        float3_t albedo;
        float    reflectivity = 0.0f; // >0 = metal
        float    fuzz         = 0.0f; // metal fuzz radius
        float    ior          = 0.0f; // >1 = dielectric
    };

    std::mt19937 engine(42);
    std::function<float()> random = std::bind(std::uniform_real_distribution<float>(), engine);

    std::vector<SphereInfo> spheres;

    spheres.push_back({float3_t(0.0f, -1000.0f, 0.0f), 1000.0f, float3_t(0.5f, 0.5f, 0.5f)});

    for (int i = -11; i < 11; ++i) {
        for (int j = -11; j < 11; ++j) {
            const float chooseMat = random();
            const float cz = static_cast<float>(j) + 0.9f * random();
            const float cx = static_cast<float>(i) + 0.9f * random();
            const float3_t center(cx, 0.2f, cz);

            if (length(center - float3_t(4.0f, 0.2f, 0.0f)) > 0.9f) {
                float3_t albedo;
                if (chooseMat < 0.8f) {
                    const float b = random() * random();
                    const float g = random() * random();
                    const float r = random() * random();
                    albedo = float3_t(r, g, b);
                    spheres.push_back({center, 0.2f, albedo, 0.0f, 0.0f});
                } else if (chooseMat < 0.95f) {
                    const float fuzziness = 0.5f * random();
                    const float b = 0.5f * (1.0f + random());
                    const float g = 0.5f * (1.0f + random());
                    const float r = 0.5f * (1.0f + random());
                    albedo = float3_t(r, g, b);
                    spheres.push_back({center, 0.2f, albedo, 1.0f, fuzziness, 0.0f});
                } else {
                    albedo = float3_t(1.0f, 1.0f, 1.0f);
                    spheres.push_back({center, 0.2f, albedo, 0.0f, 0.0f, 1.5f});
                }
            }
        }
    }

    spheres.push_back({float3_t( 0.0f, 1.0f, 0.0f), 1.0f, float3_t(0.7f, 0.7f, 1.0f), 0.0f, 0.0f, 1.5f}); // glass
    spheres.push_back({float3_t(-4.0f, 1.0f, 0.0f), 1.0f, float3_t(0.4f, 0.2f, 0.1f), 0.0f, 0.0f}); // lambertian
    spheres.push_back({float3_t( 4.0f, 1.0f, 0.0f), 1.0f, float3_t(0.7f, 0.6f, 0.5f), 1.0f, 0.0f}); // metal

    std::vector<Mesh*> meshes;
    meshes.reserve(spheres.size());
    for (auto& s : spheres) {
        auto* mesh = Mesh::CreateSphere(s.center, s.radius, s.albedo, Geomrtry::Sphere);
        if (s.ior > 0.0f) {
            mesh->setAllIOR(s.ior);
            mesh->setAllMaterialModel(MaterialDielectric);
        } else if (s.reflectivity > 0.0f) {
            mesh->setAllReflectivity(s.reflectivity);
            mesh->setAllFuzz(s.fuzz);
            mesh->setAllMaterialModel(MaterialMetallic);
        } else {
            mesh->setAllMaterialModel(MaterialLambertian);
        }
        meshes.push_back(mesh);
    }

    Scene* scene = new Scene(meshes);

    scene->camera_pos   = float3_t(13.0f, 2.0f, 3.0f);
    scene->camera_front = normalize(float3_t(0.0f, 0.0f, 0.0f) - float3_t(13.0f, 2.0f, 3.0f));
    scene->camera_fov   = 5.671f; // 20° FOV: 1/tan(10°)

    scene->light_pos     = float3_t(0.0f, 10.0f, 0.0f);
    scene->light_color   = float3_t(1.0f, 1.0f, 1.0f);
    scene->ambient_color = float3_t(0.5f, 0.6f, 0.7f);
    scene->background_color = float3_t(0.5f, 0.7f, 1.0f);
    return scene;
}

Scene* SceneList::Ship() {
    auto s_model = resolve_path("assets/karimSchooner.obj", ASSETS_PATHS);
    auto* shipMesh = new Mesh(s_model, Geomrtry::Ship);

    const mat4_t T = mat4_t::Translate(float3_t(173.0f, -9.0f, -377.0f))
        * mat4_t::Scale(100.0f)
        * mat4_t::RotateY(75.0f * PI / 180.0f);
    shipMesh->setTransform(T);

    Scene* scene = new Scene({shipMesh});

    scene->camera_pos   = float3_t(378.0f, 278.0f, 500.0f);
    scene->camera_front = normalize(float3_t(-200.0f, 0.0f, -500.0f));
    scene->camera_fov   = 2.1445f;  // 50° vertical FOV: 1/tan(25°)

    scene->light_pos        = float3_t(0.0f, 0.0f, 0.0f);
    scene->background_color = float3_t(0.5f, 0.7f, 1.0f);
    return scene;
}

Scene* SceneList::Chestnut() {
    auto s_model = resolve_path(std::string("assets/") + "Chestnut/chestnut.obj", ASSETS_PATHS);
    auto* chestnutMesh = new Mesh(s_model, Geomrtry::Chestnut, false);

    const mat4_t T = mat4_t::RotateX(-75.0f * PI / 180.0f);
    chestnutMesh->setTransform(T);

    Scene* scene = new Scene({chestnutMesh});

    scene->camera_pos   = float3_t(20.0f, 0.0f, 100.0f);
    scene->camera_front = normalize(float3_t(0.0f, 100.0f, -100.0f));
    scene->camera_fov   = 2.1445f;  // 50° vertical FOV: 1/tan(25°)

    scene->light_pos        = float3_t(0.0f, 0.0f, 0.0f);
    scene->background_color = float3_t(0.5f, 0.7f, 1.0f);
    return scene;
}

const std::vector<std::pair<std::string, std::function<Scene* (void)>>> SceneList::AllScenes = {
    {"cornellbox", CornellBox},
    {"bunny", Bunny},
    {"sponza", Sponza},
    {"carnival", Carnival},
    {"chestnut", Chestnut},
    {"ship", Ship},
    {"rtiow", RayTracingInOneWeekend}
};