#include "bvh.h"
#include "kdtree.h"
#include "treelet.h"
#include <utility>
#include <cmath>
#include <iostream>
// bin count for binned BVH building
#define BINS 8

// BVH class implementation

BVH::BVH(const tri_t *triData, const float3_t *centroids, uint32_t triCount, bvh_node_t *bvh_nodes, bvh_quantized_node_t *bvh_qnodes, uint32_t *triIndices, uint32_t offset) {
  offset_ = offset;
  bvhNodes_ = bvh_nodes;
  bvhQNodes_ = bvh_qnodes;
  centroids_ = centroids;
  triCount_ = triCount;
  triData_ = triData;
  triIndices_ = triIndices;
  
  this->build();
  //visualize(bvhNodes_);
  this->quantize();
}

BVH::~BVH() {
  //--
}

void BVH::build() {
  // Recursive build staring at the root node
  bvh_node_t &root = bvhNodes_[nodeCount_++];
  root.leftFirst = 0;
  root.triCount = triCount_;
  this->subdivide(root);
}

// void BVH::initializeNode(bvh_node_t &node, uint32_t first, uint32_t count) {
//   node.leftFirst = first;
//   node.triCount = count;

//   float3_t centroidMin, centroidMax;
//   this->updateNodeBounds(node, &centroidMin, &centroidMax);
//   if (count > 1) {
//     this->subdivide(node, centroidMin, centroidMax);
//   }
// }

void BVH::subdivide(bvh_node_t &node) {
  this->updateNodeBounds(node);
  if(node.triCount <= 1){
    return;
  }


  std::vector<bvh_node_t> clusters;
  clusters.push_back(node);
  int nMax = 4;
  float totalCost = node.calculateNodeCost();

  while (clusters.size() < nMax){
   
    Split bestSplit;
    float bestCost = 99999.0f;
    int bestIdx = -1;
    for (int i = 0; i < clusters.size(); ++i) {
       
      Split s = findBestSplitPlane(clusters[i]);
      
      if (s.cost < bestCost) {
        bestCost = s.cost; 
        bestSplit = s; 
        bestIdx = i;
      }
    }

    if(bestIdx < 0) return;

    totalCost -= clusters[bestIdx].calculateNodeCost();
    totalCost += bestCost;
    uint32_t leftCount = partitionTriangles(clusters[bestIdx], bestSplit);
    uint32_t rightCount = clusters[bestIdx].triCount - leftCount;


    if (leftCount == 0 || rightCount == 0) {
      std::cout << "Edge case" << std::endl;
      return;
    }

    
    bvh_node_t L, R;
    L.leftFirst = clusters[bestIdx].leftFirst;
    L.triCount = leftCount;
    R.leftFirst = clusters[bestIdx].leftFirst + leftCount;
    R.triCount = rightCount;

    clusters[bestIdx] = L;
    clusters.push_back(R);

  }

  if(totalCost >= node.calculateNodeCost())
    return;

  uint32_t Child0Idx = nodeCount_++;
  uint32_t Child1Idx = nodeCount_++;
  uint32_t Child2Idx = nodeCount_++;
  uint32_t Child3Idx = nodeCount_++;

  bvh_node_t &Child0 = bvhNodes_[Child0Idx];
  bvh_node_t &Child1 = bvhNodes_[Child1Idx];
  bvh_node_t &Child2 = bvhNodes_[Child2Idx];
  bvh_node_t &Child3 = bvhNodes_[Child3Idx];

  Child0.leftFirst = clusters[0].leftFirst;
  Child0.triCount  = clusters[0].triCount;
  Child1.leftFirst = clusters[1].leftFirst;
  Child1.triCount  = clusters[1].triCount;
  Child2.leftFirst = clusters[2].leftFirst;
  Child2.triCount  = clusters[2].triCount;
  Child3.leftFirst = clusters[3].leftFirst;
  Child3.triCount  = clusters[3].triCount;

  subdivide(Child0);
  subdivide(Child1);
  subdivide(Child2);
  subdivide(Child3);
  // update parent's child nodes
  node.leftFirst = Child0Idx;
  node.triCount = 0; // mark as parent node
}

uint32_t BVH::partitionTriangles(const bvh_node_t &node, const Split &split) const {
  float scale = BINS / (node.centroidMax[split.axis] - node.centroidMin[split.axis]);
  uint32_t *triPtr = triIndices_ + node.leftFirst;

  uint32_t i = 0;
  uint32_t j = node.triCount - 1;

  while (i <= j) {
    uint32_t triIdx = triPtr[i];
    auto &centroid = centroids_[triIdx];
    uint32_t bin = clamp(int((centroid[split.axis] - node.centroidMin[split.axis]) * scale), 0, BINS - 1);
    if (bin < split.pos) {
      i++;
    } else {
      std::swap(triPtr[i], triPtr[j--]);
    }
  }

  return i;
}

Split BVH::findBestSplitPlane(const bvh_node_t &node) const {
  //float bestCost = LARGE_FLOAT;
  Split bestSplit;
  bestSplit.cost = LARGE_FLOAT;
  for (uint32_t a = 0; a < 3; a++) {
    float boundsMin = node.centroidMin[a], boundsMax = node.centroidMax[a];
    if (boundsMin == boundsMax)
      continue;

    // populate the bins
    float scale = BINS / (boundsMax - boundsMin);
    float leftCountArea[BINS - 1], rightCountArea[BINS - 1];
    int leftSum = 0, rightSum = 0;

    struct Bin {
      AABB bounds;
      int triCount = 0;
    } bin[BINS];

    for (uint32_t i = 0; i < node.triCount; i++) {
      auto triIdx = triIndices_[node.leftFirst + i];
      auto &triangle = triData_[triIdx];
      auto &centroid = centroids_[triIdx];
      int binIdx = (int)((centroid[a] - boundsMin) * scale);
      binIdx = std::max(0, std::min((int)BINS - 1, binIdx));

      bin[binIdx].triCount++;
      bin[binIdx].bounds.grow(triangle.v0);
      bin[binIdx].bounds.grow(triangle.v1);
      bin[binIdx].bounds.grow(triangle.v2);
    }

    // gather data for the 7 planes between the 8 bins
    AABB leftBox, rightBox;
    for (int i = 0; i < BINS - 1; i++) {
      leftSum += bin[i].triCount;
      leftBox.grow(bin[i].bounds);
      leftCountArea[i] = leftSum * leftBox.area();
      rightSum += bin[BINS - 1 - i].triCount;
      rightBox.grow(bin[BINS - 1 - i].bounds);
      rightCountArea[BINS - 2 - i] = rightSum * rightBox.area();
    }

    // calculate SAH cost for the 7 planes
    scale = (boundsMax - boundsMin) / BINS;
    for (int i = 0; i < BINS - 1; i++) {
      const float planeCost = leftCountArea[i] + rightCountArea[i];
      if (planeCost < bestSplit.cost) {
        bestSplit.axis = a;
        bestSplit.pos = i + 1;
        bestSplit.cost = planeCost;
      }
    }
  }
  return bestSplit;
}

void BVH::updateNodeBounds(bvh_node_t &node) const {
  node.aabbMin = float3_t(LARGE_FLOAT);
  node.aabbMax = float3_t(-LARGE_FLOAT);
  auto centroid_min = float3_t(LARGE_FLOAT);
  auto centroid_max = float3_t(-LARGE_FLOAT);
  for (uint32_t first = node.leftFirst, i = 0; i < node.triCount; i++) {
    uint32_t triIdx = triIndices_[first + i];
    auto &tri = triData_[triIdx];
    node.aabbMin = fminf(node.aabbMin, tri.v0);
    node.aabbMin = fminf(node.aabbMin, tri.v1);
    node.aabbMin = fminf(node.aabbMin, tri.v2);
    node.aabbMax = fmaxf(node.aabbMax, tri.v0);
    node.aabbMax = fmaxf(node.aabbMax, tri.v1);
    node.aabbMax = fmaxf(node.aabbMax, tri.v2);
    auto &centroid = centroids_[triIdx];
    centroid_min = fminf(centroid_min, centroid);
    centroid_max = fmaxf(centroid_max, centroid);
  }
  node.centroidMin = centroid_min;
  node.centroidMax = centroid_max;
}

void BVH::quantize(){
  std::cout << "Quantization starts ... " << std::endl;
  for(int i=0; i<nodeCount_; i++){
    bvh_node_t node = bvhNodes_[i];

    bvh_quantized_node_t qNode;
    // uint32_t l = node.leftFirst + offset_;
    // uint32_t r = l + 1;
    qNode.leftRight = node.leftFirst + offset_; //(r << 16) | l;
    qNode.leafIdx = node.triCount;
    qNode.origin = node.aabbMin;

    //uint8_t
    qNode.ex = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.x - node.aabbMin.x) / 255.0f)));
    qNode.ey = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.y - node.aabbMin.y) / 255.0f)));
    qNode.ez = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.z - node.aabbMin.z) / 255.0f)));
    qNode.imask = 0;

    if(!node.isLeaf()){
      uint32_t c0 = node.leftFirst;
      uint32_t c1 = c0 + 1;
      uint32_t c2 = c1 + 1;
      uint32_t c3 = c2 + 1;
      bvh_node_t Child0 = bvhNodes_[c0];
      bvh_node_t Child1 = bvhNodes_[c1];
      bvh_node_t Child2 = bvhNodes_[c2];
      bvh_node_t Child3 = bvhNodes_[c3];

      child_data_t QChild0, QChild1, QChild2, QChild3;
      QChild0.meta = 0;
      
      QChild0.qaabb[0] = static_cast<uint8_t>(std::floor((Child0.aabbMin.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
      QChild0.qaabb[1] = static_cast<uint8_t>(std::floor((Child0.aabbMin.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
      QChild0.qaabb[2] = static_cast<uint8_t>(std::floor((Child0.aabbMin.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

      QChild0.qaabb[3] = static_cast<uint8_t>(std::ceil((Child0.aabbMax.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
      QChild0.qaabb[4] = static_cast<uint8_t>(std::ceil((Child0.aabbMax.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
      QChild0.qaabb[5] = static_cast<uint8_t>(std::ceil((Child0.aabbMax.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

      qNode.children[0] = QChild0;
      
      //------------------------------
      QChild1.meta = 0;
      
      QChild1.qaabb[0] = static_cast<uint8_t>(std::floor((Child1.aabbMin.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
      QChild1.qaabb[1] = static_cast<uint8_t>(std::floor((Child1.aabbMin.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
      QChild1.qaabb[2] = static_cast<uint8_t>(std::floor((Child1.aabbMin.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

      QChild1.qaabb[3] = static_cast<uint8_t>(std::ceil((Child1.aabbMax.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
      QChild1.qaabb[4] = static_cast<uint8_t>(std::ceil((Child1.aabbMax.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
      QChild1.qaabb[5] = static_cast<uint8_t>(std::ceil((Child1.aabbMax.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

      qNode.children[1] = QChild1;

      //------------------------------
      QChild2.meta = 0;
      
      QChild2.qaabb[0] = static_cast<uint8_t>(std::floor((Child2.aabbMin.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
      QChild2.qaabb[1] = static_cast<uint8_t>(std::floor((Child2.aabbMin.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
      QChild2.qaabb[2] = static_cast<uint8_t>(std::floor((Child2.aabbMin.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

      QChild2.qaabb[3] = static_cast<uint8_t>(std::ceil((Child2.aabbMax.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
      QChild2.qaabb[4] = static_cast<uint8_t>(std::ceil((Child2.aabbMax.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
      QChild2.qaabb[5] = static_cast<uint8_t>(std::ceil((Child2.aabbMax.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

      qNode.children[2] = QChild2;

      //------------------------------
      QChild3.meta = 0;
      
      QChild3.qaabb[0] = static_cast<uint8_t>(std::floor((Child3.aabbMin.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
      QChild3.qaabb[1] = static_cast<uint8_t>(std::floor((Child3.aabbMin.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
      QChild3.qaabb[2] = static_cast<uint8_t>(std::floor((Child3.aabbMin.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

      QChild3.qaabb[3] = static_cast<uint8_t>(std::ceil((Child3.aabbMax.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
      QChild3.qaabb[4] = static_cast<uint8_t>(std::ceil((Child3.aabbMax.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
      QChild3.qaabb[5] = static_cast<uint8_t>(std::ceil((Child3.aabbMax.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

      qNode.children[3] = QChild3;
    }

    bvhQNodes_[i] = qNode;
  }
  std::cout << "Quantization ends ... " << nodeCount_ << std::endl;
}

// TLAS implementation

TLAS::TLAS(const std::vector<BVH *> &bvh_list, const blas_node_t *blas_nodes, bvh_quantized_node_t *bvh_qnodes)
    : bvh_list_(bvh_list) {
  bvhQNodes_ = bvh_qnodes;
  blas_nodes_ = blas_nodes;
  blasCount_ = bvh_list.size();
  nodeCount_ = 2 * blasCount_ - 1;
  // allocate TLAS nodes
  tlasNodes_.resize(nodeCount_);
  nodeIndices_.resize(blasCount_);
  triCounts_.resize(blasCount_);
}

TLAS::~TLAS() {
  //--
}

void TLAS::build() {
  if (blasCount_ == 0)
    return;

  // Initialize leaf nodes
  for (uint32_t i = 0; i < blasCount_; ++i) {
    auto &bvh = bvh_list_[i];
    auto &blas_node = blas_nodes_[i];

    // calculate world-space bounds using the new matrix
    auto &aabbMin = bvh->aabbMin();
    auto &aabbMax = bvh->aabbMax();
    AABB bounds;
    for (int c = 0; c < 8; ++c) {
      float3_t pos(c & 1 ? aabbMax.x : aabbMin.x,
                   c & 2 ? aabbMax.y : aabbMin.y,
                   c & 4 ? aabbMax.z : aabbMin.z);
      bounds.grow(TransformPosition(pos, blas_node.transform));
    }

    tlasNodes_[i].aabbMin = bounds.bmin;
    tlasNodes_[i].aabbMax = bounds.bmax;
    tlasNodes_[i].blasIdx = i;
    tlasNodes_[i].setLeftRight(0, 0); // leaf node

    triCounts_[i] = bvh->triCount();
    nodeIndices_[i] = i;
  }

  uint32_t currentInternalNodeIndex = blasCount_;
  rootIndex_ = buildRecursive(0, blasCount_ - 1, currentInternalNodeIndex);
  this->quantize();
}

uint32_t TLAS::buildRecursive(uint32_t start, uint32_t end, uint32_t &currentInternalNodeIndex) {
  if (start == end) {
    return nodeIndices_[start]; // Leaf node
  }

  // Compute current AABB
  float3_t aabbMin = tlasNodes_[nodeIndices_[start]].aabbMin;
  float3_t aabbMax = tlasNodes_[nodeIndices_[start]].aabbMax;
  for (uint32_t i = start + 1; i <= end; ++i) {
    auto &node = tlasNodes_[nodeIndices_[i]];
    aabbMin = fminf(aabbMin, node.aabbMin);
    aabbMax = fmaxf(aabbMax, node.aabbMax);
  }

  // Determine best split axis and position using SAH
  int splitAxis = 0;
  float splitPos = 0.0f;
  float bestCost = std::numeric_limits<float>::infinity();

  float3_t extent = aabbMax - aabbMin;
  for (uint32_t axis = 0; axis < 3; ++axis) {
    if (extent[axis] <= 0)
      continue;

    float binWidth = extent[axis] / BINS;
    for (uint32_t i = 1; i < BINS; ++i) {
      float candidatePos = aabbMin[axis] + i * binWidth;

      // Calculate SAH cost
      uint32_t leftTris = 0, rightTris = 0;
      float3_t leftMin(LARGE_FLOAT), leftMax(-LARGE_FLOAT);
      float3_t rightMin(LARGE_FLOAT), rightMax(-LARGE_FLOAT);

      for (uint32_t j = start; j <= end; ++j) {
        const tlas_node_t &node = tlasNodes_[nodeIndices_[j]];
        float centroid = (node.aabbMin[axis] + node.aabbMax[axis]) / 2;
        if (centroid < candidatePos) {
          leftMin = fminf(leftMin, node.aabbMin);
          leftMax = fmaxf(leftMax, node.aabbMax);
          leftTris += triCounts_[node.blasIdx];
        } else {
          rightMin = fminf(rightMin, node.aabbMin);
          rightMax = fmaxf(rightMax, node.aabbMax);
          rightTris += triCounts_[node.blasIdx];
        }
      }
      if (leftTris == 0 || rightTris == 0)
        continue; // no valid split

      // Compute SAH cost
      float leftArea = surfaceArea(leftMin, leftMax);
      float rightArea = surfaceArea(rightMin, rightMax);
      float cost = leftArea * leftTris + rightArea * rightTris;
      if (cost < bestCost) {
        bestCost = cost;
        splitAxis = axis;
        splitPos = candidatePos;
      }
    }
  }

  // Fallback to median split if SAH failed
  if (bestCost == std::numeric_limits<float>::infinity()) {
    splitAxis = (extent.x > extent.y) ? ((extent.x > extent.z) ? 0 : 2) : ((extent.y > extent.z) ? 1 : 2);
    // Compute median centroid along splitAxis
    std::vector<float> centroids;
    for (uint32_t i = start; i <= end; ++i) {
        const auto &node = tlasNodes_[nodeIndices_[i]];
        float centroid = (node.aabbMin[splitAxis] + node.aabbMax[splitAxis]) * 0.5f;
        centroids.push_back(centroid);
    }
    std::sort(centroids.begin(), centroids.end());
    splitPos = centroids[centroids.size() / 2]; // median
  }

  // Partition the primitives based on the best split
  uint32_t mid = partition(start, end, splitAxis, splitPos);
  if (mid == start || mid == end) {
    mid = (start + end) / 2;
  }

  // Recursively build left and right subtrees
  uint32_t leftChild = buildRecursive(start, mid, currentInternalNodeIndex);
  uint32_t rightChild = buildRecursive(mid + 1, end, currentInternalNodeIndex);

  // Create internal node
  uint32_t nodeIndex = currentInternalNodeIndex++;
  auto &node = tlasNodes_[nodeIndex];
  node.setLeftRight(leftChild, rightChild);
  node.aabbMin = aabbMin;
  node.aabbMax = aabbMax;

  return nodeIndex;
}

uint32_t TLAS::partition(int start, int end, int axis, float splitPos) {
  int left = start;
  int right = end;

  while (left <= right) {
    while (left <= end) {
      auto &node = tlasNodes_[nodeIndices_[left]];
      float centroid = (node.aabbMin[axis] + node.aabbMax[axis]) / 2;
      if (centroid < splitPos)
        left++;
      else
        break;
    }
    while (right >= start) {
      auto &node = tlasNodes_[nodeIndices_[right]];
      float centroid = (node.aabbMin[axis] + node.aabbMax[axis]) / 2;
      if (centroid >= splitPos)
        right--;
      else
        break;
    }
    if (left < right) {
      std::swap(nodeIndices_[left], nodeIndices_[right]);
      left++;
      right--;
    }
  }

  // All elements < splitPos → force split at last element
  if (right < start)
    return end;

  // All elements >= splitPos → force split at first element
  if (left > end)
    return start;

  // Return partition point
  return right;
}

void TLAS::quantize(){
  std::cout << "TLAS Quantization starts ... " << std::endl;
  for(int i=0; i<nodeCount_; i++){
    tlas_node_t node = tlasNodes_[i];

    bvh_quantized_node_t qNode;
    qNode.leftRight = node.leftRight;
    qNode.leafIdx = node.blasIdx;
    qNode.origin = node.aabbMin;

    //uint8_t
    qNode.ex = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.x - node.aabbMin.x) / 255.0f)));
    qNode.ey = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.y - node.aabbMin.y) / 255.0f)));
    qNode.ez = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.z - node.aabbMin.z) / 255.0f)));
    qNode.imask = 1;

    if(!node.isLeaf()){
      uint32_t left = node.leftRight & 0xFFFF;
      uint32_t right = node.leftRight >> 16;
      tlas_node_t leftNode = tlasNodes_[left];
      tlas_node_t rightNode = tlasNodes_[right];

      child_data_t leftChild;
      leftChild.meta = 0;
      
      leftChild.qaabb[0] = static_cast<uint8_t>(std::floor((leftNode.aabbMin.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
      leftChild.qaabb[1] = static_cast<uint8_t>(std::floor((leftNode.aabbMin.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
      leftChild.qaabb[2] = static_cast<uint8_t>(std::floor((leftNode.aabbMin.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

      leftChild.qaabb[3] = static_cast<uint8_t>(std::ceil((leftNode.aabbMax.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
      leftChild.qaabb[4] = static_cast<uint8_t>(std::ceil((leftNode.aabbMax.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
      leftChild.qaabb[5] = static_cast<uint8_t>(std::ceil((leftNode.aabbMax.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

      qNode.children[0] = leftChild;
      
      child_data_t rightChild;
      rightChild.meta = 0;
      
      rightChild.qaabb[0] = static_cast<uint8_t>(std::floor((rightNode.aabbMin.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
      rightChild.qaabb[1] = static_cast<uint8_t>(std::floor((rightNode.aabbMin.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
      rightChild.qaabb[2] = static_cast<uint8_t>(std::floor((rightNode.aabbMin.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

      rightChild.qaabb[3] = static_cast<uint8_t>(std::ceil((rightNode.aabbMax.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
      rightChild.qaabb[4] = static_cast<uint8_t>(std::ceil((rightNode.aabbMax.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
      rightChild.qaabb[5] = static_cast<uint8_t>(std::ceil((rightNode.aabbMax.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

      qNode.children[1] = rightChild;
    }

    bvhQNodes_[i] = qNode;
  }
  std::cout << "TLAS Quantization ends ... " << nodeCount_ << std::endl;
}