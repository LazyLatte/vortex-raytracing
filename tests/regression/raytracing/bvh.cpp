#include "bvh.h"
#include "kdtree.h"
#include <utility>
#include <cmath>
#include <iostream>
#include <algorithm>

// bin count for binned BVH building
#define BINS 8
// BVH class implementation

BVH::BVH(tri_t *triData, float3_t *centroids, uint32_t triCount, bvh_node_t *bvh_nodes, uint32_t *triIndices) {
  bvhNodes_   = bvh_nodes;
  centroids_  = centroids;
  triCount_   = triCount;
  triData_    = triData;
  triIndices_ = triIndices;
  this->build();
}

BVH::BVH(AABB *aabbData, float3_t *centroids, uint32_t count, bvh_node_t *bvh_nodes, uint32_t *indices) {
  bvhNodes_   = bvh_nodes;
  centroids_  = centroids;
  triCount_   = count;
  aabbData_   = aabbData;
  triIndices_ = indices;
  this->build();
}

BVH::~BVH() {}

void BVH::build() {
  // Recursive build staring at the root node
  bvh_node_t &root = bvhNodes_[nodeCount_++];
  root.leftFirst = 0;
  root.triCount = triCount_;
  this->updateNodeBounds(root);
  this->subdivide(root);
}

void BVH::subdivide(bvh_node_t &node) {

  if(node.triCount <= MAX_LEAF_PRIMITIVES){
    return;
  }
  
  struct ClusterInfo {
    bvh_node_t node;
    Split bestSplit;
    float costReduction;
  };

  auto evaluateCluster = [&](ClusterInfo &c) {
    if (c.node.triCount <= MAX_LEAF_PRIMITIVES) {
      c.costReduction = -1.0f; // Cannot split
      return;
    }

    c.bestSplit = findBestSplitPlane(c.node);
    
    if (c.bestSplit.cost == std::numeric_limits<float>::infinity()) {
      c.costReduction = -1.0f;
    } else {
      // Ensure c.node.calculateNodeCost() returns (AABB surface area * triCount)
      c.costReduction = c.node.calculateNodeCost() - c.bestSplit.cost;
    }
  };

  ClusterInfo clusters[BVH_WIDTH];
  int clusterCount = 0;

  clusters[clusterCount++] = {node, {}, -1.0f};
  evaluateCluster(clusters[0]);

  while (clusterCount < BVH_WIDTH) {
    float bestDelta = 0.0f;
    int bestIdx = -1;

    for (int i = 0; i < clusterCount; i++) {
      if (clusters[i].costReduction > bestDelta) {
        bestDelta = clusters[i].costReduction;
        bestIdx = i;
      }
    }

    if (bestIdx < 0) break; // No improving split found across all clusters

    ClusterInfo best = clusters[bestIdx];
    uint32_t leftCount = partitionTriangles(best.node, best.bestSplit);
    uint32_t rightCount = best.node.triCount - leftCount;
    assert(leftCount > 0 && rightCount > 0);

    ClusterInfo L, R;
    L.node.leftFirst = best.node.leftFirst;
    L.node.triCount = leftCount;
    R.node.leftFirst = best.node.leftFirst + leftCount;
    R.node.triCount = rightCount;

    this->updateNodeBounds(L.node);
    this->updateNodeBounds(R.node);

    evaluateCluster(L);
    evaluateCluster(R);

    clusters[bestIdx] = L;
    clusters[clusterCount++] = R;
  }

  if (clusterCount == 1) {
    // Leaf node logic: it couldn't be split efficiently
    return; 
  }

  uint32_t childIndices[BVH_WIDTH];
  for (int i = 0; i < clusterCount; i++) {
    childIndices[i] = nodeCount_++;
  }

  for (int i = 0; i < clusterCount; i++) {
    bvh_node_t &childNode = bvhNodes_[childIndices[i]];
    childNode = clusters[i].node; 
    subdivide(childNode);
  }

  // Mark current node as internal
  node.triCount = 0; 
  node.leftFirst = childIndices[0];
  node.childCount = clusterCount;
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
      uint32_t triIdx = triIndices_[node.leftFirst + i];
      auto &centroid = centroids_[triIdx];
      int binIdx = (int)((centroid[a] - boundsMin) * scale);
      binIdx = std::max(0, std::min((int)BINS - 1, binIdx));

      bin[binIdx].triCount++;
      if (aabbData_) {
        auto &aabb = aabbData_[triIdx];
        bin[binIdx].bounds.grow(aabb.bmin);
        bin[binIdx].bounds.grow(aabb.bmax);
      } else {
        auto &tri = triData_[triIdx];
        bin[binIdx].bounds.grow(tri.v0);
        bin[binIdx].bounds.grow(tri.v1);
        bin[binIdx].bounds.grow(tri.v2);
      }
      
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
    if (aabbData_) {
      auto &aabb = aabbData_[triIdx];
      node.aabbMin = fminf(node.aabbMin, aabb.bmin);
      node.aabbMax = fmaxf(node.aabbMax, aabb.bmax);
    } else {
      auto &tri = triData_[triIdx];
      node.aabbMin = fminf(node.aabbMin, tri.v0);
      node.aabbMin = fminf(node.aabbMin, tri.v1);
      node.aabbMin = fminf(node.aabbMin, tri.v2);
      node.aabbMax = fmaxf(node.aabbMax, tri.v0);
      node.aabbMax = fmaxf(node.aabbMax, tri.v1);
      node.aabbMax = fmaxf(node.aabbMax, tri.v2);
    }
    auto &centroid = centroids_[triIdx];
    centroid_min = fminf(centroid_min, centroid);
    centroid_max = fmaxf(centroid_max, centroid);
  }
  node.centroidMin = centroid_min;
  node.centroidMax = centroid_max;
}

// TLAS implementation

TLAS::TLAS(const std::vector<BVH *> &bvh_list, const blas_node_t *blas_nodes) : bvh_list_(bvh_list) {
  blas_nodes_ = blas_nodes;
  blasCount_ = bvh_list.size();
  nodeCount_ = 2 * blasCount_ - 1;
  tlasLeaves_.resize(blasCount_);
  tlasNodes_.resize(nodeCount_);
  nodeIndices_.resize(blasCount_);
}

TLAS::~TLAS() {
  //--
}

void TLAS::build() {
  if (blasCount_ == 0)
    return;

  // Initialize leaf nodes and index array
  for (uint32_t i = 0; i < blasCount_; ++i) {
    auto &bvh = bvh_list_[i];
    auto &blas_node = blas_nodes_[i];

    auto &aabbMin = bvh->aabbMin();
    auto &aabbMax = bvh->aabbMax();
    AABB bounds;
    for (int c = 0; c < 8; ++c) {
      float3_t pos(c & 1 ? aabbMax.x : aabbMin.x,
                   c & 2 ? aabbMax.y : aabbMin.y,
                   c & 4 ? aabbMax.z : aabbMin.z);
      bounds.grow(TransformPosition(pos, blas_node.transform.toMat4()));
    }

    tlasLeaves_[i].leftFirst = i; // BLAS index
    tlasLeaves_[i].aabbMin = bounds.bmin;
    tlasLeaves_[i].aabbMax = bounds.bmax;
    tlasLeaves_[i].triCount = bvh->triCount();
    tlasLeaves_[i].childCount = 0;
    nodeIndices_[i] = i;
  }

  tlas_node_t &root = tlasNodes_[nodeIndex_++];
  updateNode(root, 0, blasCount_ - 1);
  buildRecursive(root);
}

void TLAS::buildRecursive(tlas_node_t &node) {

  if (node.start == node.end) {
    node = tlasLeaves_[nodeIndices_[node.start]];
    return;
  }

  struct ClusterInfo {
    tlas_node_t node;
    Split bestSplit;
    float costReduction;
  };

  auto evaluateCluster = [&](ClusterInfo &c) {
    if (c.node.start == c.node.end) {
      c.costReduction = -1.0f; // Cannot split
      return;
    }
    c.bestSplit = findBestSplitPlane(c.node);
    if (c.bestSplit.cost == std::numeric_limits<float>::infinity()) {
      c.costReduction = -1.0f;
    } else {
      c.costReduction = c.node.calculateNodeCost() - c.bestSplit.cost;
    }
  };

  ClusterInfo clusters[BVH_WIDTH];
  int clusterCount = 0;

  clusters[clusterCount++] = {node, {}, -1.0f};
  evaluateCluster(clusters[0]);

  while (clusterCount < BVH_WIDTH) {
    float bestDelta = 0.0f;
    int bestIdx = -1;

    for (int i = 0; i < clusterCount; i++) {
      if (clusters[i].costReduction > bestDelta) {
        bestDelta = clusters[i].costReduction;
        bestIdx = i;
      }
    }

    if (bestIdx < 0) break; // No improving split found across all clusters

    ClusterInfo best = clusters[bestIdx];
    uint32_t mid = partition(best.node.start, best.node.end, best.bestSplit.axis, best.bestSplit.pos);

    if (mid < best.node.start || mid >= best.node.end) break; // degenerate split

    ClusterInfo L, R;
    updateNode(L.node, best.node.start, mid);
    updateNode(R.node, mid + 1, best.node.end);

    evaluateCluster(L);
    evaluateCluster(R);

    clusters[bestIdx] = L;
    clusters[clusterCount++] = R;
  }

  if (clusterCount == 1) {
    // SAH found no improving split; fall back to median split
    uint32_t childIndices[2] = {nodeIndex_++, nodeIndex_++};
    uint32_t mid = (node.start + node.end) / 2;
    tlas_node_t &leftChild  = tlasNodes_[childIndices[0]];
    tlas_node_t &rightChild = tlasNodes_[childIndices[1]];
    updateNode(leftChild, node.start, mid);
    updateNode(rightChild, mid + 1, node.end);
    buildRecursive(leftChild);
    buildRecursive(rightChild);
    node.leftFirst = childIndices[0];
    node.childCount = 2;
    return;
  }

  uint32_t childIndices[BVH_WIDTH];
  for (int i = 0; i < clusterCount; i++) {
    childIndices[i] = nodeIndex_++;
  }

  for (int i = 0; i < clusterCount; i++) {
    tlas_node_t &child = tlasNodes_[childIndices[i]];
    child = clusters[i].node;
    buildRecursive(child);
  }

  node.leftFirst = childIndices[0];
  node.childCount = clusterCount;
}

uint32_t TLAS::partition(int start, int end, int axis, float splitPos) {
  int left = start;
  int right = end;

  while (left <= right) {
    while (left <= end) {
      auto &leaf = tlasLeaves_[nodeIndices_[left]];
      float centroid = (leaf.aabbMin[axis] + leaf.aabbMax[axis]) / 2;
      if (centroid < splitPos)
        left++;
      else
        break;
    }
    while (right >= start) {
      auto &leaf = tlasLeaves_[nodeIndices_[right]];
      float centroid = (leaf.aabbMin[axis] + leaf.aabbMax[axis]) / 2;
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

  if (right < start)
    return end;

  if (left > end)
    return start;

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
        const tlas_node_t &leaf = tlasLeaves_[nodeIndices_[j]];
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
    count += tlasLeaves_[nodeIndices_[i]].triCount;
  }
  node.triCount = count;
}

void TLAS::updateNodeBounds(tlas_node_t &node) const {
  AABB bounds;
  for (uint32_t i = node.start; i <= node.end; ++i) {
    bounds.grow(tlasLeaves_[nodeIndices_[i]].aabbMin);
    bounds.grow(tlasLeaves_[nodeIndices_[i]].aabbMax);
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