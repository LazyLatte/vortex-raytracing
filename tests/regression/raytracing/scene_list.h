#pragma once
#include "scene.h"
#include <functional>
#include <string>
#include <tuple>
#include <vector>
#include <unordered_map>

class SceneList final {
public:
    static Scene* CornellBox();
    static Scene* Sponza();
    static Scene* Spring();
	static const std::vector<std::pair<std::string, std::function<Scene* (void)>>> AllScenes;
};
