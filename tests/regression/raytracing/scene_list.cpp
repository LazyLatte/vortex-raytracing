#include "scene_list.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

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
    std::vector<Mesh*> meshes(1);
    auto s_model = resolve_path(std::string("assets/") + "CornellBox/CornellBox-Original.obj", ASSETS_PATHS);
    meshes[0] = new Mesh(s_model.c_str(), (uint32_t)Geomrtry::CornellBox);

    // create scene
    Scene* scene = new Scene(meshes);
    scene->camera_pos = float3_t(0.0f, 1.0f, 2.0f);
    scene->camera_front = float3_t(0.0f, 0.0f, -1.0f);
    scene->camera_fov = 1.428f;

    // scene->light_pos = float3_t(0, 10, -10);
    // scene->light_color = float3_t(1, 1, 1.0);
    // scene->ambient_color = float3_t(0.4f, 0.4f, 0.4f);
    // scene->background_color = float3_t(0.4f, 0.35f, 0.25f);
    return scene;
}

Scene* SceneList::Sponza(){
    std::vector<Mesh*> meshes(1);
    auto s_model = resolve_path(std::string("assets/") + "Sponza/sponza.obj", ASSETS_PATHS);
    meshes[0] = new Mesh(s_model.c_str(), (uint32_t)Geomrtry::Sponza);

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

        auto mesh = new Mesh(entry.path(), (uint32_t)Geomrtry::Spring);
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


const std::vector<std::pair<std::string, std::function<Scene* (void)>>> SceneList::AllScenes = {
  {"cornellbox", CornellBox},
	{"sponza", Sponza},
  {"spring", Spring}
};