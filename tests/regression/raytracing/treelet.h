#pragma once
#include "common.h"
#include <queue>
#include <iostream>
#include <fstream>
#include <string>

typedef std::queue<ray_t>  treelet_t;
void treelet_cost_calculation(bvh_node_t *bvhBuffer, uint32_t rootIdx, float *bestCost);
void treelet_assignment(bvh_node_t *bvhBuffer, uint32_t &treeletID, float *bestCost);
void visualize(bvh_node_t* root, const std::string& filename = "tree.dot");