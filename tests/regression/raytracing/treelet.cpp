#include "treelet.h"
#include <algorithm>
#include <set>
#include <limits>
#include <sstream>
#include <iomanip>
#include <random>

#define BVH_STACK_SIZE 256
#define MAX_QUEUE_NUM 8
#define MAX_TREELET_FOOTPRINT 4096
#define INF 3.4028235e+38

//Stage 1
void treelet_cost_calculation(bvh_node_t *bvhBuffer, uint32_t rootIdx, float *bestCost){
    const bvh_node_t &root = bvhBuffer[rootIdx];
    if(!root.isLeaf()){
        treelet_cost_calculation(bvhBuffer, root.leftFirst, bestCost);
        treelet_cost_calculation(bvhBuffer, root.leftFirst + 1, bestCost);
    }
  
    uint32_t footprint = sizeof(bvh_node_t);

    std::set<uint32_t> cut;
    cut.insert(rootIdx);
    uint32_t bytesRemaining = MAX_TREELET_FOOTPRINT;
    bestCost[rootIdx] = std::numeric_limits<float>::max();
    float eps = 0.0;

    while(1){
        int32_t bestNodeIdx = -1;
        float bestScore = std::numeric_limits<float>::lowest();

        for(uint32_t n : cut){
            const bvh_node_t &node = bvhBuffer[n];
            if(footprint <= bytesRemaining){
                float gain = surfaceArea(node.aabbMin, node.aabbMax) + eps;
                //float price = std::min(node.triCount * 36, bytesRemaining);
                float score = gain;
                //float score = gain / price;
                if(score > bestScore){
                    bestNodeIdx = n;
                    bestScore = score;
                }
            }
        }

        if(bestNodeIdx < 0)
            break;

        const bvh_node_t &bestNode = bvhBuffer[bestNodeIdx];
        if(!bestNode.isLeaf()){
            uint32_t l = bestNode.leftFirst;
            uint32_t r = l + 1;

            cut.insert(l);
            cut.insert(r);
        }

        cut.erase(bestNodeIdx);
        bytesRemaining -= footprint;

        float total_cut_cost = 0;
        for (uint32_t n : cut) {
            total_cut_cost += bestCost[n];
        }

        float cost = (surfaceArea(root.aabbMin, root.aabbMax)+ eps) + total_cut_cost;
        bestCost[rootIdx] = std::min(bestCost[rootIdx], cost);
    }
}

//Stage 2
void treelet_assignment(bvh_node_t *bvhBuffer, uint32_t &treeletID, float *bestCost){
  std::queue<uint32_t> Q;
  Q.push(0);
  uint32_t footprint = sizeof(bvh_node_t);
  //uint32_t treeletID = -1;
  while (!Q.empty()) {
    treeletID++;
    uint32_t rootIdx = Q.front(); Q.pop();
    const bvh_node_t &root = bvhBuffer[rootIdx];
    
    //start
    std::set<uint32_t> cut;
    cut.insert(rootIdx);
    uint32_t bytesRemaining = MAX_TREELET_FOOTPRINT;
    float eps = 0.0;

    float cost = 0;

    while(1){
        int32_t bestNodeIdx = -1;
        float bestScore = std::numeric_limits<float>::lowest();

        for(uint32_t n : cut){
            const bvh_node_t &node = bvhBuffer[n];
            if(footprint <= bytesRemaining){
                float gain = surfaceArea(node.aabbMin, node.aabbMax) + eps;
                //float price = std::min(node.triCount * 36, bytesRemaining);
                float score = gain;
                //float score = gain / price;
                if(score > bestScore){
                    bestNodeIdx = n;
                    bestScore = score;
                }
            }
        }
        if(bestNodeIdx < 0){
            std::cout << "Can't enlarge the treelet: " << rootIdx << " " << cost << " " <<  bestCost[rootIdx] << std::endl;
            break;
        }
            

        bvh_node_t &bestNode = bvhBuffer[bestNodeIdx];
        if(!bestNode.isLeaf()){
            uint32_t l = bestNode.leftFirst;
            uint32_t r = l + 1;

            cut.insert(l);
            cut.insert(r);
        }

        cut.erase(bestNodeIdx);
        bestNode.treeletID = treeletID;
        bytesRemaining -= footprint;

        float total_cut_cost = 0;
        for (uint32_t n : cut) {
            total_cut_cost += bestCost[n];
        }

        float cost = (surfaceArea(root.aabbMin, root.aabbMax)+ eps) + total_cut_cost;
        if(cost == bestCost[rootIdx]){
            //std::cout << "Treelet: ID=" << treeletID << " IDX: " << rootIdx << " Cost: " << cost << std::endl;
            for (uint32_t n : cut) {
                Q.push(n);
            }
            break;
        }
    }
    //end
  }
        
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

// Export function
void exportDOT(bvh_node_t* bvhBuffer, uint32_t rootIdx, std::ofstream& out, const std::vector<std::string>& colors) {
    const bvh_node_t &root = bvhBuffer[rootIdx];
    std::string color = colors[root.treeletID % colors.size()];

    out << "  " << nodeName(&bvhBuffer[rootIdx]) << " [style=filled, fillcolor=\"" << color << "\", label=\"\"];\n";

    if(!root.isLeaf()){
        uint32_t left = root.leftFirst;
        uint32_t right = left + 1;
        out << "  " << nodeName(&bvhBuffer[rootIdx]) << " -> " << nodeName(&bvhBuffer[left]) << ";\n";
        exportDOT(bvhBuffer, left, out, colors);
        out << "  " << nodeName(&bvhBuffer[rootIdx]) << " -> " << nodeName(&bvhBuffer[right]) << ";\n";
        exportDOT(bvhBuffer, right, out, colors);
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

