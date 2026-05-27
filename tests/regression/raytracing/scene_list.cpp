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
    scene->camera_pos = float3_t(0.0f, 1.0f, 3.88288288288f);
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
    scene->camera_pos   = float3_t(0.0f, 1.0f, 3.88288288288f);
    scene->camera_front = float3_t(0.0f, 0.0f, -1.0f);
    scene->camera_fov   = 2.74747741945f;

    scene->light_pos        = float3_t(-0.005f, 1.98f, -0.003f);
    scene->background_color = float3_t(0.0f, 0.0f, 0.0f);
    return scene;
}

Scene* SceneList::Sponza(){
    std::vector<Mesh*> meshes(1);
    auto s_model = resolve_path(std::string("assets/") + "Sponza/sponza.obj", ASSETS_PATHS);
    meshes[0] = new Mesh(s_model.c_str(), Geomrtry::Sponza);

    // create scene
    Scene* scene = new Scene(meshes);
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

Scene* SceneList::Spring(){
    namespace fs = std::filesystem;

    const std::string dir = resolve_path("assets/Spring", ASSETS_PATHS);
    std::vector<Mesh*> meshes;
    float3_t cam_pos{}, cam_front{0, 0, -1};

    for (const auto& entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();

        if (entry.path().extension() == ".camera") {
            float3_t cam_target{};
            std::ifstream f(entry.path());
            f >> cam_pos.x >> cam_pos.y >> cam_pos.z;
            f >> cam_target.x >> cam_target.y >> cam_target.z;
            cam_front = normalize(cam_target - cam_pos);
            continue;
        }

        if (entry.path().extension() != ".obj")
            continue;
        if (name.find("Dirt_grass") != std::string::npos)
            continue;

        auto mesh = new Mesh(entry.path(), Geomrtry::Spring);
        if (mesh->tri().empty()) {
            delete mesh;
            continue;
        }

        if (name.find("spring_body") != std::string::npos)
            mesh->setAllDiffuse({223.0f/256, 175.0f/256, 171.0f/256});
        else if (name.find("stitches") != std::string::npos)
            mesh->setAllDiffuse({92.0f/256, 64.0f/256, 51.0f/256});
        else if (name.find("spring_jacket") != std::string::npos)
            mesh->setAllDiffuse({163.0f/256, 67.0f/256, 42.0f/256});
        else if (name.find("spring_pants") != std::string::npos)
            mesh->setAllDiffuse({92.0f/256, 74.0f/256, 101.0f/256});
        else if (name.find("spring_boots") != std::string::npos)
            mesh->setAllDiffuse({150.0f/256, 106.0f/256, 86.0f/256});
        else if (name.find("spring_hairband") != std::string::npos)
            mesh->setAllDiffuse({69.0f/256, 23.0f/256, 8.0f/256});
        else if (name.find("spring_hair") != std::string::npos)
            mesh->setAllDiffuse({108.0f/256, 86.0f/256, 99.0f/256});
        else if (name.find("spring_scarf") != std::string::npos || name.find("spring_pullover") != std::string::npos)
            mesh->setAllDiffuse({114.0f/256, 76.0f/256, 64.0f/256});

        meshes.push_back(mesh);
    }

    Scene* scene = new Scene(meshes);
    scene->camera_pos   = cam_pos;
    scene->camera_front = cam_front;
    scene->camera_fov   = 5.671f;  // 20° FOV: 1/tan(10°)

    scene->light_pos     = float3_t(0.0f, 0.0f, 0.0f);
    scene->light_color   = float3_t(1.0f, 1.0f, 1.0f);
    scene->ambient_color = float3_t(0.15f, 0.15f, 0.15f);
    return scene;
}


Scene* SceneList::RtInOneWeekend() {
    struct SphereInfo {
        float3_t center;
        float    radius;
        float3_t albedo;
    };

    std::mt19937 engine(42);
    std::function<float()> random = std::bind(std::uniform_real_distribution<float>(), engine);

    std::vector<SphereInfo> spheres;

    // Ground sphere
    spheres.push_back({float3_t(0.0f, -1000.0f, 0.0f), 1000.0f, float3_t(0.5f, 0.5f, 0.5f)});

    // Random small spheres — random() call order matches the reference exactly
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
                } else if (chooseMat < 0.95f) {
                    const float fuzz = 0.5f * random(); (void)fuzz;
                    const float b = 0.5f * (1.0f + random());
                    const float g = 0.5f * (1.0f + random());
                    const float r = 0.5f * (1.0f + random());
                    albedo = float3_t(r, g, b);
                } else {
                    albedo = float3_t(1.0f, 1.0f, 1.0f);
                }
                spheres.push_back({center, 0.2f, albedo});
            }
        }
    }

    // Three main spheres
    spheres.push_back({float3_t( 0.0f, 1.0f, 0.0f), 1.0f, float3_t(0.9f, 0.9f, 0.9f)}); // glass
    spheres.push_back({float3_t(-4.0f, 1.0f, 0.0f), 1.0f, float3_t(0.4f, 0.2f, 0.1f)}); // lambertian
    spheres.push_back({float3_t( 4.0f, 1.0f, 0.0f), 1.0f, float3_t(0.7f, 0.6f, 0.5f)}); // metal

    const std::string spherePath = resolve_path("assets/sphere.obj", ASSETS_PATHS);

    std::vector<Mesh*> meshes;
    meshes.reserve(spheres.size());

    for (auto& s : spheres) {
        // sphere.obj is a unit sphere (radius 0.5) centred at the origin.
        // Scale by 2*r to reach the desired radius, then translate to the sphere centre.
        const mat4_t T = mat4_t::Translate(s.center) * mat4_t::Scale(2.0f * s.radius);
        auto* mesh = new Mesh(spherePath, Geomrtry::RtInOneWeekend);
        mesh->setTransform(T);
        mesh->setAllDiffuse(s.albedo);
        meshes.push_back(mesh);
    }

    Scene* scene = new Scene(meshes);

    scene->camera_pos   = float3_t(13.0f, 2.0f, 3.0f);
    scene->camera_front = normalize(float3_t(0.0f, 0.0f, 0.0f) - float3_t(13.0f, 2.0f, 3.0f));
    scene->camera_fov   = 5.671f; // 20° FOV: 1/tan(10°)

    scene->light_pos     = float3_t(0.0f, 10.0f, 0.0f);
    scene->light_color   = float3_t(1.0f, 1.0f, 1.0f);
    scene->ambient_color = float3_t(0.5f, 0.6f, 0.7f);

    return scene;
}

Scene* SceneList::RtInOneWeekendProc() {
    struct SphereInfo {
        float3_t center;
        float    radius;
        float3_t albedo;
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
                } else if (chooseMat < 0.95f) {
                    const float fuzz = 0.5f * random(); (void)fuzz;
                    const float b = 0.5f * (1.0f + random());
                    const float g = 0.5f * (1.0f + random());
                    const float r = 0.5f * (1.0f + random());
                    albedo = float3_t(r, g, b);
                } else {
                    albedo = float3_t(1.0f, 1.0f, 1.0f);
                }
                spheres.push_back({center, 0.2f, albedo});
            }
        }
    }

    spheres.push_back({float3_t( 0.0f, 1.0f, 0.0f), 1.0f, float3_t(0.9f, 0.9f, 0.9f)});
    spheres.push_back({float3_t(-4.0f, 1.0f, 0.0f), 1.0f, float3_t(0.4f, 0.2f, 0.1f)});
    spheres.push_back({float3_t( 4.0f, 1.0f, 0.0f), 1.0f, float3_t(0.7f, 0.6f, 0.5f)});

    std::vector<Mesh*> meshes;
    meshes.reserve(spheres.size());
    for (auto& s : spheres)
        meshes.push_back(Mesh::CreateSphere(s.center, s.radius, s.albedo, Geomrtry::Sphere));

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

const std::vector<std::pair<std::string, std::function<Scene* (void)>>> SceneList::AllScenes = {
    {"cornellbox", CornellBox},
    {"bunny", Bunny},
    {"sponza", Sponza},
    {"carnival", Carnival},
    {"spring", Spring},
    {"rtiow", RtInOneWeekend},
    {"rtiow-proc", RtInOneWeekendProc}
};