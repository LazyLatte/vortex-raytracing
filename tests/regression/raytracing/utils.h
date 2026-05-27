#pragma once
#include "common.h"
#include <queue>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <set>
#include <limits>
#include <sstream>
#include <iomanip>
#include <random>

uint32_t max_depth(bvh_node_t* bvhBuffer, uint32_t idx) {
    const bvh_node_t& node = bvhBuffer[idx];
    if(node.isLeaf()) return 0;

    uint32_t md = 0;
    for(uint32_t i=0; i<BVH_WIDTH; i++){
        if(i < node.childCount){
            uint32_t d = max_depth(bvhBuffer, node.leftFirst + i);
            md = std::max(md, d);
        }
    }
    return md + 1;
}

uint32_t max_depth(const std::vector<tlas_node_t>& tlasBuffer, uint32_t idx) {
    const tlas_node_t& node = tlasBuffer[idx];
    if(node.isLeaf()) return 0;

    uint32_t md = 0;
    for(uint32_t i=0; i<BVH_WIDTH; i++){
        if(i < node.childCount){
            uint32_t d = max_depth(tlasBuffer, node.leftFirst + i);
            md = std::max(md, d);
        }
    }
    return md + 1;
}

std::string hslToHex(float h, float s, float l) {
    auto hue2rgb = [](float p, float q, float t) {
        if(t < 0) t += 1.0f;
        if(t > 1) t -= 1.0f;
        if(t < 1.0/6) return p + (q - p) * 6.0f * t;
        if(t < 1.0/2) return q;
        if(t < 2.0/3) return p + (q - p) * (2.0f/3.0f - t) * 6.0f;
        return p;
    };

    float r, g, b;
    h /= 360; // normalize hue

    if(s == 0) {
        r = g = b = l; // achromatic
    } else {
        float q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        float p = 2 * l - q;
        r = hue2rgb(p, q, h + 1.0/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1.0/3);
    }

    std::ostringstream oss;
    oss << "#" << std::hex << std::setfill('0') << std::setw(2) << int(r * 255)
        << std::setw(2) << int(g * 255) << std::setw(2) << int(b * 255);
    return oss.str();
}

std::vector<std::string> generateColorPalette(int count) {
    std::vector<std::string> colors;
    for (int i = 0; i < count; ++i) {
        float h = (360.0f / count) * i;  // evenly spaced hue
        float s = 0.7f; // saturation
        float l = 0.5f; // lightness
        colors.push_back(hslToHex(h, s, l));
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(colors.begin(), colors.end(), rng);
    

    return colors;
}


std::string nodeName(bvh_node_t* node) {
    return "node" + std::to_string(reinterpret_cast<uintptr_t>(node));
}

void exportDOT(bvh_node_t* bvhBuffer, uint32_t rootIdx, std::ofstream& out, const std::vector<std::string>& colors) {
    const bvh_node_t &root = bvhBuffer[rootIdx];
    //std::string color = colors[root.treeletID % colors.size()];
    std::string color = colors[0];
    out << "  " << nodeName(&bvhBuffer[rootIdx]) << " [style=filled, fillcolor=\"" << color << "\", label=\"\"];\n";

    if(!root.isLeaf()){
        uint32_t left = root.leftFirst;
        for(int i=0; i<4; i++){
            uint32_t childIdx = root.leftFirst + i;
            if(i < root.childCount){
                out << "  " << nodeName(&bvhBuffer[rootIdx]) << " -> " << nodeName(&bvhBuffer[childIdx]) << ";\n";
                exportDOT(bvhBuffer, childIdx, out, colors);
            }
        }
    }
}

// Wrapper
void visualize(bvh_node_t* root, const std::string& filename) {
    std::ofstream out(filename);
    out << "digraph G {\n";
    out << "  node [shape=circle, fontname=\"Arial\"];\n";
    std::vector<std::string> colors = generateColorPalette(100); 
    exportDOT(root, 0, out, colors);
    out << "}\n";
    out.close();
    std::cout << "DOT file written to " << filename << std::endl;
}