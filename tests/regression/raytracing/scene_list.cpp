#include "scene_list.h"
#include <iostream>
#include <fstream>
#include <sstream>

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
    meshes[0] = new Mesh(s_model.c_str());

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
    meshes[0] = new Mesh(s_model.c_str());

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

const std::vector<std::pair<std::string, std::function<Scene* (void)>>> SceneList::AllScenes = {
  {"cornellbox", CornellBox},
	{"sponza", Sponza}
};