#include "bvh.h"
#include "kdtree.h"
#include "treelet.h"
#include <utility>
#include <cmath>
#include <iostream>
// bin count for binned BVH building
#define BINS 8
// BVH class implementation

BVH::BVH(tri_t *triData, float3_t *centroids, uint32_t triCount, bvh_node_t *bvh_nodes, bvh_quantized_node_t *bvh_qnodes, uint32_t *triIndices, tri_ex_t *triEx, uint32_t tri_offset) {
  bvhNodes_ = bvh_nodes;
  bvhQNodes_ = bvh_qnodes;
  centroids_ = centroids;
  triCount_ = triCount;
  triData_ = triData;
  triIndices_ = triIndices;
  triEx_ = triEx;
  tri_offset_ = tri_offset;
  std::cout << "Start build" <<std::endl;
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

void BVH::subdivide(bvh_node_t &node) {
  //MAX num triangles per leaf = ???
  this->updateNodeBounds(node);

  if(node.triCount <= 1){
    return;
  }
  
  std::vector<bvh_node_t> clusters;
  clusters.push_back(node);

  while (clusters.size() < BVH_WIDTH){
    Split bestSplit{};
    float bestDelta = 0.0f;
    int bestIdx = -1;
    //std::cout << "test 1" << std::endl;
    for (int i = 0; i < clusters.size(); ++i) {
      auto& c = clusters[i];
      if (c.triCount <= 1) continue;
      
      Split s = findBestSplitPlane(c);
      
      // All triangles are in the same bin => no partition happens 
      if(s.cost == std::numeric_limits<float>::infinity()) continue;

      float delta = c.calculateNodeCost() - s.cost;
      if (delta > bestDelta) {
        bestDelta = delta; 
        bestSplit = s; 
        bestIdx = i;
      }
    }
    //std::cout << "test 2" << std::endl;
    if(bestIdx < 0) break;  //No improving split

    uint32_t leftCount = partitionTriangles(clusters[bestIdx], bestSplit);
    uint32_t rightCount = clusters[bestIdx].triCount - leftCount;
    //if(leftCount == 0 || rightCount == 0) break;
    //std::cout << leftCount << " " << rightCount << std::endl;
    assert(leftCount != 0 && rightCount != 0);

    bvh_node_t L, R;
    L.leftFirst = clusters[bestIdx].leftFirst;
    L.triCount = leftCount;
    R.leftFirst = clusters[bestIdx].leftFirst + leftCount;
    R.triCount = rightCount;

    clusters[bestIdx] = L;
    clusters.push_back(R);
  }

  if(clusters.size() == 1){
    //std::cout << "Leaf TriCount > 1" << std::endl;
    return;
  }

  uint32_t childIndices[BVH_WIDTH];
  for(int i=0; i<clusters.size(); i++){
    childIndices[i] = nodeCount_++;
  }

  for(int i=0; i<clusters.size(); i++){
      bvh_node_t &childNode = bvhNodes_[childIndices[i]];
      childNode.leftFirst = clusters[i].leftFirst;
      childNode.triCount = clusters[i].triCount;
      subdivide(childNode);
  }

  node.triCount = 0; // mark as internal node
  node.leftFirst = childIndices[0];
  node.childCount = clusters.size();
}

uint32_t BVH::partitionTriangles(const bvh_node_t &node, const Split &split) const {
  float scale = BINS / (node.centroidMax[split.axis] - node.centroidMin[split.axis]);
  //uint32_t *triPtr = triIndices_ + node.leftFirst;

  uint32_t i = 0;
  uint32_t j = node.triCount - 1;

  while (i <= j) {
    //uint32_t triIdx = triPtr[i];
    auto &centroid = centroids_[node.leftFirst + i];
    uint32_t bin = clamp(int((centroid[split.axis] - node.centroidMin[split.axis]) * scale), 0, BINS - 1);
    if (bin < split.pos) {
      i++;
    } else {
      //std::swap(triPtr[i], triPtr[j--]);
      std::swap(triData_[node.leftFirst + i], triData_[node.leftFirst + j]);
      std::swap(triEx_[node.leftFirst + i], triEx_[node.leftFirst + j]);
      std::swap(centroids_[node.leftFirst + i], centroids_[node.leftFirst + j]);
      j--;
    }
  }
  return i;
}

Split BVH::findBestSplitPlane(const bvh_node_t &node) const {

  Split bestSplit;

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
      // triIdx = triIndices_[node.leftFirst + i];
      auto &triangle = triData_[node.leftFirst + i];
      auto &centroid = centroids_[node.leftFirst + i];
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
      leftCountArea[i] = leftSum > 0 ? (leftSum * leftBox.area()) : std::numeric_limits<float>::infinity();
      rightSum += bin[BINS - 1 - i].triCount;
      rightBox.grow(bin[BINS - 1 - i].bounds);
      rightCountArea[BINS - 2 - i] = rightSum > 0 ? (rightSum * rightBox.area()) : std::numeric_limits<float>::infinity();
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
    //uint32_t triIdx = triIndices_[first + i];
    auto &tri = triData_[first + i];
    node.aabbMin = fminf(node.aabbMin, tri.v0);
    node.aabbMin = fminf(node.aabbMin, tri.v1);
    node.aabbMin = fminf(node.aabbMin, tri.v2);
    node.aabbMax = fmaxf(node.aabbMax, tri.v0);
    node.aabbMax = fmaxf(node.aabbMax, tri.v1);
    node.aabbMax = fmaxf(node.aabbMax, tri.v2);
    auto &centroid = centroids_[first + i];
    centroid_min = fminf(centroid_min, centroid);
    centroid_max = fmaxf(centroid_max, centroid);
  }
  node.centroidMin = centroid_min;
  node.centroidMax = centroid_max;
}

void BVH::quantize(){
  std::cout << "BVH Quantization starts ... " << std::endl;
  for(int i=0; i<nodeCount_; i++){
    bvh_node_t node = bvhNodes_[i];
    bvh_quantized_node_t &qNode = bvhQNodes_[i];

    qNode.leftFirst = node.leftFirst; 
    qNode.leafIdx = node.triCount;
    qNode.origin = node.aabbMin;

    //uint8_t
    qNode.ex = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.x - node.aabbMin.x) / 255.0f)));
    qNode.ey = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.y - node.aabbMin.y) / 255.0f)));
    qNode.ez = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.z - node.aabbMin.z) / 255.0f)));

    // Right now, this is just to identify if node is toplevel or not
    // Could be storing childnode types (internal/leaf), then traversal stack entry should include a isLeaf bit.
    qNode.imask = 0; 

    if(!node.isLeaf()){
      for(int k=0; k<BVH_WIDTH; k++){
        child_data_t qChild;

        if(k < node.childCount){
          bvh_node_t child = bvhNodes_[node.leftFirst + k];

          qChild.meta = 1; //Do we need meta info, or just a single valid bit?

          qChild.qaabb[0] = static_cast<uint8_t>(std::floor((child.aabbMin.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
          qChild.qaabb[1] = static_cast<uint8_t>(std::floor((child.aabbMin.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
          qChild.qaabb[2] = static_cast<uint8_t>(std::floor((child.aabbMin.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

          qChild.qaabb[3] = static_cast<uint8_t>(std::ceil((child.aabbMax.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
          qChild.qaabb[4] = static_cast<uint8_t>(std::ceil((child.aabbMax.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
          qChild.qaabb[5] = static_cast<uint8_t>(std::ceil((child.aabbMax.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

        }else{
          qChild.meta = 0;
        }
        qNode.children[k] = qChild;
      }
    }else{
      // if(node.triCount > 1){
      //   std::cout << node.triCount << std::endl;
      // }
      qNode.leftFirst += tri_offset_;
    }
  }
  std::cout << "BVH Quantization ends ... (#node=" << nodeCount_ << ")" << std::endl;
}

// TLAS implementation

TLAS::TLAS(const std::vector<BVH *> &bvh_list, const blas_node_t *blas_nodes) : bvh_list_(bvh_list) {
  blas_nodes_ = blas_nodes;
  blasCount_ = bvh_list.size();
  nodeCount_ = 2 * blasCount_ - 1;
  // allocate TLAS nodes
  tlasLeaves_.resize(blasCount_);
  tlasNodes_.resize(nodeCount_);
  tlasQNodes_.resize(nodeCount_);
  // nodeIndices_.resize(blasCount_);
  // triCounts_.resize(blasCount_);

  //nodeIndex_ = blasCount_;
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

    tlasLeaves_[i].aabbMin = bounds.bmin;
    tlasLeaves_[i].aabbMax = bounds.bmax;
    tlasLeaves_[i].blasIdx = i;
    tlasLeaves_[i].setLeftRight(0, 0); // leaf node
    tlasLeaves_[i].triCount = bvh->triCount();
    
    //triCounts_[i] = bvh->triCount();
    //nodeIndices_[i] = i;
  }

  tlas_node_t &root = tlasNodes_[nodeIndex_++];
  updateNode(root, 0, blasCount_ - 1);
  buildRecursive(root);
  this->quantize();

}

void TLAS::buildRecursive(tlas_node_t &node) {

  if (node.start == node.end) {
    node = tlasLeaves_[node.start];
    return;
  }

  std::vector<tlas_node_t> clusters;
  clusters.push_back(node);

  while (clusters.size() < BVH_WIDTH){
    int bestIdx = -1;
    Split bestSplit{};
    float bestDelta = 0.0f;

    for (int i = 0; i < (int)clusters.size(); ++i) {
        auto& c = clusters[i];
        if (c.start == c.end) continue;

        //std::cout << "findBestSplitPlane: " <<c.start << " " << c.end << std::endl;
        Split s = findBestSplitPlane(c);
        if (s.cost == std::numeric_limits<float>::infinity())
            continue;

        float leafCost = c.calculateNodeCost();
        float delta = leafCost - s.cost; // >0 = good
        if (delta > bestDelta) {
            bestDelta = delta;
            bestSplit = s;
            bestIdx = i;
        }
    }

    if (bestIdx < 0) break; // no improving split

    auto& c = clusters[bestIdx];
    uint32_t mid = partition(c.start, c.end, bestSplit.axis, bestSplit.pos);

    if (mid < c.start || mid >= c.end) break; // degenerate split → stop

    tlas_node_t L, R;
    updateNode(L, c.start, mid);
    updateNode(R, mid + 1, c.end);

    clusters[bestIdx] = L;
    clusters.push_back(R);
  }
  //std::cout << clusters.size() << std::endl;
  uint32_t childIndices[BVH_WIDTH];
  if (clusters.size() == 1) {
    childIndices[0] = nodeIndex_++;
    childIndices[1] = nodeIndex_++;

    uint32_t mid = (node.start + node.end) / 2;

    tlas_node_t &leftChild  = tlasNodes_[childIndices[0]];
    tlas_node_t &rightChild = tlasNodes_[childIndices[1]];
    updateNode(leftChild, node.start, mid);
    updateNode(rightChild, mid+1, node.end);
    buildRecursive(leftChild);
    buildRecursive(rightChild);
    //return makeInternalNode({l, r}); // 2-wide

    // Fallback to median split if SAH failed
    // if (bestSplit.cost == std::numeric_limits<float>::infinity()) {
    //   bestSplit.axis = (extent.x > extent.y) ? ((extent.x > extent.z) ? 0 : 2) : ((extent.y > extent.z) ? 1 : 2);
    //   // Compute median centroid along splitAxis
    //   std::vector<float> centroids;
    //   for (uint32_t i = start; i <= end; ++i) {
    //       const auto &node = tlasNodes_[nodeIndices_[i]];
    //       float centroid = (node.aabbMin[bestSplit.axis] + node.aabbMax[bestSplit.axis]) * 0.5f;
    //       centroids.push_back(centroid);
    //   }
    //   std::sort(centroids.begin(), centroids.end());
    //   bestSplit.pos = centroids[centroids.size() / 2]; // median
    // }
  }else{
    for(int i=0; i<clusters.size(); i++){
      childIndices[i] = nodeIndex_++;
    }

    for(int i=0; i<clusters.size(); i++){
      tlas_node_t &child  = tlasNodes_[childIndices[i]];
      updateNode(child, clusters[i].start, clusters[i].end);
      buildRecursive(child);
    }
  }
  // for(int i=0; i<clusters.size(); i++){
  //   std::cout << childIndices[i] << " ";
  // }

  // std::cout << std::endl;

  node.leftFirst = childIndices[0];
  node.blasIdx = UINT32_MAX;
  node.childCount = clusters.size() == 1 ? 2 : clusters.size();
  //---------------

  // // Partition the primitives based on the best split
  // uint32_t mid = partition(start, end, splitAxis, splitPos);
  // if (mid == start || mid == end) {
  //   mid = (start + end) / 2;
  // }

  // // Recursively build left and right subtrees
  // uint32_t leftChild = buildRecursive(start, mid, currentInternalNodeIndex);
  // uint32_t rightChild = buildRecursive(mid + 1, end, currentInternalNodeIndex);

  // // Create internal node
  // uint32_t nodeIndex = currentInternalNodeIndex++;
  // auto &node = tlasNodes_[nodeIndex];
  // node.setLeftRight(leftChild, rightChild);
  // node.aabbMin = aabbMin;
  // node.aabbMax = aabbMax;

  // return nodeIndex;
}

uint32_t TLAS::partition(int start, int end, int axis, float splitPos) {
  int left = start;
  int right = end;

  while (left <= right) {
    while (left <= end) {
      auto &node = tlasLeaves_[left];
      float centroid = (node.aabbMin[axis] + node.aabbMax[axis]) / 2;
      if (centroid < splitPos)
        left++;
      else
        break;
    }
    while (right >= start) {
      auto &node = tlasLeaves_[right];
      float centroid = (node.aabbMin[axis] + node.aabbMax[axis]) / 2;
      if (centroid >= splitPos)
        right--;
      else
        break;
    }
    if (left < right) {
      std::swap(tlasLeaves_[left], tlasLeaves_[right]);
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

Split TLAS::findBestSplitPlane(tlas_node_t &node) const {
  Split bestSplit;
  bestSplit.axis = 0;
  bestSplit.pos = 0;
  bestSplit.cost = std::numeric_limits<float>::infinity();

  uint32_t start = node.start, end = node.end;
  float3_t aabbMin = node.aabbMin, aabbMax = node.aabbMax;

  float3_t extent = aabbMax - aabbMin;

  // Determine best split axis and position using SAH
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
        const tlas_node_t &leaf = tlasLeaves_[j];
        float centroid = (leaf.aabbMin[axis] + leaf.aabbMax[axis]) / 2;
        if (centroid < candidatePos) {
          leftMin = fminf(leftMin, leaf.aabbMin);
          leftMax = fmaxf(leftMax, leaf.aabbMax);
          leftTris += leaf.triCount;
        } else {
          rightMin = fminf(rightMin, leaf.aabbMin);
          rightMax = fmaxf(rightMax, leaf.aabbMax);
          rightTris += leaf.triCount;
        }
      }
      if (leftTris == 0 || rightTris == 0)
        continue; // no valid split

      // Compute SAH cost
      float leftArea = surfaceArea(leftMin, leftMax);
      float rightArea = surfaceArea(rightMin, rightMax);
      float cost = leftArea * leftTris + rightArea * rightTris;
      if (cost < bestSplit.cost) {
        bestSplit.cost = cost;
        bestSplit.axis = axis;
        bestSplit.pos = candidatePos;
      }
    }
  }

  return bestSplit;
}

// tlas_node_t TLAS::make_tlas_node(uint32_t start, uint32_t end){
//   tlas_node_t node;
//   node.start = start;
//   node.end = end;
//   updateNodeBounds(node);
//   updateTriCount(node);
//   return node;
// }

void TLAS::updateNode(tlas_node_t &node, uint32_t start, uint32_t end){
  node.start = start;
  node.end = end;
  updateNodeBounds(node);
  updateTriCount(node);
}

void TLAS::updateTriCount(tlas_node_t &node) const {
  uint32_t count = 0;
  for (uint32_t i = node.start; i <= node.end; ++i) {
    count += tlasLeaves_[i].triCount;
  }
  node.triCount = count;
}

void TLAS::updateNodeBounds(tlas_node_t &node) const {
  AABB bounds;
  for (uint32_t i = node.start; i <= node.end; ++i) {
    bounds.grow(tlasLeaves_[i].aabbMin);
    bounds.grow(tlasLeaves_[i].aabbMax);
  }
  node.aabbMin = bounds.bmin;
  node.aabbMax = bounds.bmax;
}

// TLAS::Cluster TLAS::make_cluster(uint32_t start, uint32_t end){
//   Cluster c;
//   c.start = start;
//   c.end = end;
//   c.bounds = this->updateNodeBounds(start, end);
//   c.triCount = this->updateTriCount(start, end);
//   return c;
// }

void TLAS::quantize(){
  std::cout << "TLAS Quantization starts ... " << std::endl;
  for(int i=0; i<nodeIndex_; i++){
    tlas_node_t node = tlasNodes_[i];
    bvh_quantized_node_t &qNode = tlasQNodes_[i];

    qNode.leftFirst = node.leftFirst;
    qNode.leafIdx = node.blasIdx;
    qNode.origin = node.aabbMin;

    //uint8_t
    qNode.ex = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.x - node.aabbMin.x) / 255.0f)));
    qNode.ey = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.y - node.aabbMin.y) / 255.0f)));
    qNode.ez = static_cast<int8_t>(std::ceil(std::log2((node.aabbMax.z - node.aabbMin.z) / 255.0f)));
    qNode.imask = 1; //TLAS node

    if(!node.isLeaf()){
      for(int k=0; k<BVH_WIDTH; k++){

        child_data_t qChild;

        if(k < node.childCount){
          tlas_node_t child = tlasNodes_[node.leftFirst + k];
          qChild.meta = 1;

          qChild.qaabb[0] = static_cast<uint8_t>(std::floor((child.aabbMin.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
          qChild.qaabb[1] = static_cast<uint8_t>(std::floor((child.aabbMin.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
          qChild.qaabb[2] = static_cast<uint8_t>(std::floor((child.aabbMin.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));

          qChild.qaabb[3] = static_cast<uint8_t>(std::ceil((child.aabbMax.x - qNode.origin.x) / std::exp2f(static_cast<float>(qNode.ex))));
          qChild.qaabb[4] = static_cast<uint8_t>(std::ceil((child.aabbMax.y - qNode.origin.y) / std::exp2f(static_cast<float>(qNode.ey))));
          qChild.qaabb[5] = static_cast<uint8_t>(std::ceil((child.aabbMax.z - qNode.origin.z) / std::exp2f(static_cast<float>(qNode.ez))));
        }else{
          qChild.meta = 0;
        }

        qNode.children[k] = qChild;
      }
    }
    std::cout << "Node " << i << ": " << qNode.leftFirst << " " << qNode.leafIdx << " " << node.childCount << std::endl;
  }
  std::cout << "TLAS Quantization ends ... (#node=" << nodeIndex_ << ", " << nodeCount_ << ")" << std::endl;
  //std::cout << "Root Idx: " << rootIndex_ << std::endl;

}