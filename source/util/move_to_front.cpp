// Copyright (c) 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "move_to_front.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace spvutils {

size_t MoveToFront::RankFromId(uint32_t id) {
  // The id was previously deprecated (i.e left its scope).
  assert(deprecated_ids_.count(id) == 0);

  const size_t old_size = GetSize();
  const auto it = id_to_node_.find(id);

  if (it == id_to_node_.end()) {
    InsertNode(CreateNode(next_timestamp_++, id));
    assert(old_size + 1 == GetSize());
    return old_size;
  }

  uint16_t target = it->second;

  uint16_t node = target;
  size_t rank = SizeOf(LeftOf(node));
  while (node) {
    if (IsRightChild(node))
      rank += 1 + SizeOf(LeftOf(ParentOf(node)));
    node = ParentOf(node);
  }

  // Update timestamp and reposition the node.
  target = RemoveNode(target);
  assert(old_size == GetSize() + 1);
  MutableTimestampOf(target) = next_timestamp_++;
  InsertNode(target);
  assert(old_size == GetSize());

  return rank;
}

uint32_t MoveToFront::IdFromRank(size_t rank) {
  const size_t old_size = GetSize();
  if (rank >= old_size) {
    assert(rank == old_size);
    const uint32_t new_id = static_cast<uint32_t>(rank + 1);
    InsertNode(CreateNode(next_timestamp_++, new_id));
    assert(old_size + 1 == GetSize());
    return new_id;
  }

  uint16_t node = root_;
  while (node) {
    const size_t left_subtree_num_nodes = SizeOf(LeftOf(node));
    if (rank == left_subtree_num_nodes) {
      // This is the node we are looking for.
      node = RemoveNode(node);
      assert(old_size == GetSize() + 1);
      MutableTimestampOf(node) = next_timestamp_++;
      InsertNode(node);
      assert(old_size == GetSize());
      return IdOf(node);
    }

    if (rank < left_subtree_num_nodes) {
      // Descend into the left subtree. The rank is still valid.
      node = LeftOf(node);
    } else {
      // Descend into the right subtree. We leave behind the left subtree and
      // the current node, adjust the |rank| accordingly.
      rank -= left_subtree_num_nodes + 1;
      node = RightOf(node);
    }
  }

  assert(0);
  return 0;
}

void MoveToFront::PrintTreeInternal(std::ostream& out, uint16_t node, size_t depth) const {
  if (!node) {
    out << "D" << depth - 1 << std::endl;
    return;
  }

  const size_t kTextFieldWidth = 10;

  std::stringstream label;
  label << IdOf(node) << "H" << HeightOf(node) << "S" << SizeOf(node);
  const size_t label_length = label.str().length();
  if (label_length < kTextFieldWidth)
    label << std::string(kTextFieldWidth - label_length, '-');

  out << label.str();

  PrintTreeInternal(out, RightOf(node), depth + 1);

  if (LeftOf(node)) {
    out << std::string(depth * kTextFieldWidth, ' ');
    PrintTreeInternal(out, LeftOf(node), depth + 1);
  }
}

void MoveToFront::InsertNode(uint16_t node) {
  if (!root_) {
    root_ = node;
    return;
  }

  uint16_t iter = root_;
  uint16_t parent = 0;

  bool right_child;

  while (iter) {
    parent = iter;
    assert(TimestampOf(iter) != TimestampOf(node));
    right_child = TimestampOf(iter) > TimestampOf(node);
    iter = right_child ? RightOf(iter) : LeftOf(iter);
  }

  assert(parent);

  MutableParentOf(node) = parent;

  if (right_child)
    MutableRightOf(parent) = node;
  else
    MutableLeftOf(parent) = node;

  // Insertion is finished. Start the balancing process.
  bool needs_rebalancing = true;
  parent = ParentOf(node);

  while (parent) {
    UpdateNode(parent);

    if (needs_rebalancing) {
      const int parent_balance = BalanceOf(parent);

      if (RightOf(parent) == node) {
        // Added node to the right subtree.
        if (parent_balance > 1) {
          // Parent is right heavy, rotate left.
          if (BalanceOf(node) < 0)
            RotateRight(node);
          parent = RotateLeft(parent);
        } else if (parent_balance == 0 || parent_balance == -1) {
          // Parent is balanced or left heavy, no need to balance further.
          needs_rebalancing = false;
        }
      } else {
        // Added node to the left subtree.
        if (parent_balance < -1) {
          // Parent is left heavy, rotate right.
          if (BalanceOf(node) > 0)
            RotateLeft(node);
          parent = RotateRight(parent);
        } else if (parent_balance == 0 || parent_balance == 1) {
          // Parent is balanced or right heavy, no need to balance further.
          needs_rebalancing = false;
        }
      }
    }

    assert(BalanceOf(parent) >= -1 && (BalanceOf(parent) <= 1));

    node = parent;
    parent = ParentOf(parent);
  }
}

uint16_t MoveToFront::RemoveNode(uint16_t node) {
  // Instead of removing the |node| find another node which is easier to
  // remove and swap them.
  if (const uint16_t scapegoat = RightestDescendantOf(LeftOf(node))) {
    std::swap(MutableIdOf(node), MutableIdOf(scapegoat));
    std::swap(MutableTimestampOf(node), MutableTimestampOf(scapegoat));
    id_to_node_[IdOf(node)] = node;
    id_to_node_[IdOf(scapegoat)] = scapegoat;
    node = scapegoat;
  }

  // node may have only one child at this point.
  assert(!RightOf(node) || !LeftOf(node));

  uint16_t parent = ParentOf(node);
  uint16_t child = RightOf(node) ? RightOf(node) : LeftOf(node);

  // Orphan node and reconnect parent and child.
  if (child)
    MutableParentOf(child) = parent;

  if (parent) {
    if (LeftOf(parent) == node)
      MutableLeftOf(parent) = child;
    else
      MutableRightOf(parent) = child;
  }

  MutableParentOf(node) = 0;
  MutableLeftOf(node) = 0;
  MutableRightOf(node) = 0;
  const uint16_t orphan = node;

  if (root_ == node)
    root_ = child;

  // Removal is finished. Start the balancing process.
  bool needs_rebalancing = true;
  node = child;

  while (parent) {
    UpdateNode(parent);

    if (needs_rebalancing) {
      const int parent_balance = BalanceOf(parent);

      if (parent_balance == 1 || parent_balance == -1) {
        // The height of the subtree was not changed.
        needs_rebalancing = false;
      } else {
        if (RightOf(parent) == node) {
          // Removed node from the right subtree.
          if (parent_balance < -1) {
            // Parent is left heavy, rotate right.
            const uint16_t sibling = LeftOf(parent);
            if (BalanceOf(sibling) > 0)
              RotateLeft(sibling);
            parent = RotateRight(parent);
          }
        } else {
          // Removed node from the left subtree.
          if (parent_balance > 1) {
            // Parent is right heavy, rotate left.
            const uint16_t sibling = RightOf(parent);
            if (BalanceOf(sibling) < 0)
              RotateRight(sibling);
            parent = RotateLeft(parent);
          }
        }
      }
    }

    assert(BalanceOf(parent) >= -1 && (BalanceOf(parent) <= 1));

    node = parent;
    parent = ParentOf(parent);
  }

  return orphan;
}

uint16_t MoveToFront::RotateLeft(const uint16_t node) {
  const uint16_t pivot = RightOf(node);

  // LeftOf(pivot) gets attached to node in place of pivot.
  MutableRightOf(node) = LeftOf(pivot);
  if (RightOf(node))
    MutableParentOf(RightOf(node)) = node;

  // Pivot gets attached to ParentOf(node) in place of node.
  MutableParentOf(pivot) = ParentOf(node);
  if (!ParentOf(node))
    root_ = pivot;
  else if (IsLeftChild(node))
    MutableLeftOf(ParentOf(node)) = pivot;
  else
    MutableRightOf(ParentOf(node)) = pivot;

  // Node is child of pivot.
  MutableLeftOf(pivot) = node;
  MutableParentOf(node) = pivot;

  UpdateNode(node);
  UpdateNode(pivot);

  return pivot;
}

uint16_t MoveToFront::RotateRight(const uint16_t node) {
  const uint16_t pivot = LeftOf(node);

  // RightOf(pivot) gets attached to node in place of pivot.
  MutableLeftOf(node) = RightOf(pivot);
  if (LeftOf(node))
    MutableParentOf(LeftOf(node)) = node;

  // Pivot gets attached to ParentOf(node) in place of node.
  MutableParentOf(pivot) = ParentOf(node);
  if (!ParentOf(node))
    root_ = pivot;
  else if (IsLeftChild(node))
    MutableLeftOf(ParentOf(node)) = pivot;
  else
    MutableRightOf(ParentOf(node)) = pivot;

  // Node is child of pivot.
  MutableRightOf(pivot) = node;
  MutableParentOf(node) = pivot;

  UpdateNode(node);
  UpdateNode(pivot);

  return pivot;
}

void MoveToFront::UpdateNode(uint16_t handle) {
  MutableSizeOf(handle) = uint16_t(
      1 + SizeOf(LeftOf(handle)) + SizeOf(RightOf(handle)));
  MutableHeightOf(handle) = uint8_t(
      1 + std::max(HeightOf(LeftOf(handle)), HeightOf(RightOf(handle))));
}

}  // namespace spvutils
