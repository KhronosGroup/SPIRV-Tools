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
  const size_t old_size = GetSize();
  const auto it = id_to_node_.find(id);

  if (it == id_to_node_.end()) {
    assert(id == next_id_);
    InsertNode(CreateNode(next_timestamp_++, next_id_++));
    assert(old_size + 1 == GetSize());
    return 0;
  }

  uint32_t target = it->second;

  uint32_t node = target;
  size_t rank = 1 + SizeOf(LeftOf(node));
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
  if (rank == 0) {
    const uint32_t new_id = next_id_++;
    InsertNode(CreateNode(next_timestamp_++, new_id));
    assert(old_size + 1 == GetSize());
    return new_id;
  }

  assert(rank <= old_size);

  uint32_t node = root_;
  while (node) {
    const size_t left_subtree_num_nodes = SizeOf(LeftOf(node));
    if (rank == left_subtree_num_nodes + 1) {
      // This is the node we are looking for.
      node = RemoveNode(node);
      assert(old_size == GetSize() + 1);
      MutableTimestampOf(node) = next_timestamp_++;
      InsertNode(node);
      assert(old_size == GetSize());
      return IdOf(node);
    }

    if (rank < left_subtree_num_nodes + 1) {
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

void MoveToFront::PrintTreeInternal(std::ostream& out, uint32_t node,
                                    size_t depth, bool print_timestamp) const {
  if (!node) {
    out << "D" << depth - 1 << std::endl;
    return;
  }

  const size_t kTextFieldWidthWithoutTimestamp = 10;
  const size_t kTextFieldWidthWithTimestamp = 14;
  const size_t text_field_width = print_timestamp ?
      kTextFieldWidthWithTimestamp : kTextFieldWidthWithoutTimestamp;

  std::stringstream label;
  label << IdOf(node) << "H" << HeightOf(node) << "S" << SizeOf(node);
  if (print_timestamp)
    label << "T" << TimestampOf(node);
  const size_t label_length = label.str().length();
  if (label_length < text_field_width)
    label << std::string(text_field_width - label_length, '-');

  out << label.str();

  PrintTreeInternal(out, RightOf(node), depth + 1, print_timestamp);

  if (LeftOf(node)) {
    out << std::string(depth * text_field_width, ' ');
    PrintTreeInternal(out, LeftOf(node), depth + 1, print_timestamp);
  }
}

void MoveToFront::InsertNode(uint32_t node) {
  assert(IsOrphan(node));
  assert(SizeOf(node) == 1);
  assert(HeightOf(node) == 1);
  assert(TimestampOf(node));

  if (!root_) {
    root_ = node;
    return;
  }

  uint32_t iter = root_;
  uint32_t parent = 0;

  // Will determine if |node| will become the right of left child after
  // insertion (but before balancing).
  bool right_child;

  // Find the node which will become |node|'s parent after insertion
  // (but before balancing).
  while (iter) {
    parent = iter;
    assert(TimestampOf(iter) != TimestampOf(node));
    right_child = TimestampOf(iter) > TimestampOf(node);
    iter = right_child ? RightOf(iter) : LeftOf(iter);
  }

  assert(parent);

  // Connect node and parent.
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

uint32_t MoveToFront::RemoveNode(uint32_t node) {
  // Instead of removing the |node| find another node which is easier to
  // remove and swap them.
  // We use the 'rightest node on the left side' as it has the following
  // properties:
  // 1. No more than one child.
  // 2. After removal of |node| it will take its place in the ranking.
  //
  // If scapegoat doesn't exist, then proceed with deleting |node|.
  if (const uint32_t scapegoat = RightestDescendantOf(LeftOf(node))) {
    std::swap(MutableIdOf(node), MutableIdOf(scapegoat));
    std::swap(MutableTimestampOf(node), MutableTimestampOf(scapegoat));
    id_to_node_[IdOf(node)] = node;
    id_to_node_[IdOf(scapegoat)] = scapegoat;
    node = scapegoat;
  }

  // |node| may have only one child at this point.
  assert(!RightOf(node) || !LeftOf(node));

  uint32_t parent = ParentOf(node);
  uint32_t child = RightOf(node) ? RightOf(node) : LeftOf(node);

  // Orphan |node| and reconnect parent and child.
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
  UpdateNode(node);
  const uint32_t orphan = node;

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
            const uint32_t sibling = LeftOf(parent);
            if (BalanceOf(sibling) > 0)
              RotateLeft(sibling);
            parent = RotateRight(parent);
          }
        } else {
          // Removed node from the left subtree.
          if (parent_balance > 1) {
            // Parent is right heavy, rotate left.
            const uint32_t sibling = RightOf(parent);
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

uint32_t MoveToFront::RotateLeft(const uint32_t node) {
  const uint32_t pivot = RightOf(node);
  assert(pivot);

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

  // Update both node and pivot. Pivot is the new parent of node, so node should
  // be updated first.
  UpdateNode(node);
  UpdateNode(pivot);

  return pivot;
}

uint32_t MoveToFront::RotateRight(const uint32_t node) {
  const uint32_t pivot = LeftOf(node);
  assert(pivot);

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

  // Update both node and pivot. Pivot is the new parent of node, so node should
  // be updated first.
  UpdateNode(node);
  UpdateNode(pivot);

  return pivot;
}

void MoveToFront::UpdateNode(uint32_t node) {
  MutableSizeOf(node) = 1 + SizeOf(LeftOf(node)) + SizeOf(RightOf(node));
  MutableHeightOf(node) =
      1 + std::max(HeightOf(LeftOf(node)), HeightOf(RightOf(node)));
}

}  // namespace spvutils
