// Copyright (c) 2025 LunarG Inc.
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

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// The remap pass is an optimization to improve compression of SPIR-V binary
// files via entropy reduction. It transforms SPIR-V to SPIR-V, remapping IDs.
// The resulting modules have an increased ID range (IDs are not as tightly
// packed around zero), but will compress better when multiple modules are
// compressed together, since the compressor's dictionary can find better cross
// module commonality. Remapping is accomplished via canonicalization. Thus,
// modules can be compressed one at a time with no loss of quality relative to
// operating on many modules at once.

// This pass should be run after most optimization passes except for
// --strip-debug because this pass will use OpName to remap IDs. i.e. Run
// --strip-debug after this pass.

class RemapIdsPass : public Pass {
 public:
  RemapIdsPass() = default;
  virtual ~RemapIdsPass() = default;

  Pass::Status Process() override;

  const char* name() const override { return "remap"; }

 private:
  // special values for ids
  static constexpr spv::Id unmapped_{spv::Id(-10000)};
  static constexpr spv::Id unused_{spv::Id(-10001)};

  // tracking for ids we want to remap
  std::vector<spv::Id> type_and_const_ids_;
  std::unordered_map<std::string, spv::Id> name_ids_;
  std::vector<spv::Id> function_ids_;
  std::vector<spv::Id> remainder_ids;

  // scans the module for ids and sets them to unmapped_
  void ScanIds();

  // functions to compute new ids
  void RemapTypeAndConst();
  spv::Id HashTypeAndConst(
      spv::Id const id) const;  // helper for RemapTypeAndConst
  void RemapNames();
  void RemapFunctions();
  spv::Id HashOpCode(
      Instruction const* const inst) const;  // helper for RemapFunctions
  void RemapRemainders();

  // applies the new ids
  void ApplyMap();

  // misc. helpers for remapping
  bool IsTypeOp(spv::Op const opCode) const;
  bool IsConstOp(spv::Op const opCode) const;
  spv::Id GetBound() const;  // all ids must satisfy 0 < id < bound.
  void UpdateBound();
  inline spv::Id NextUnusedNewId(
      spv::Id id) const;  // return next unused new id.

  // mapping from old ids to new ids e.g. new_id_[old_id] = new_id
  std::vector<spv::Id> new_id_;
  spv::Id GetNewId(spv::Id const old_id) const { return new_id_[old_id]; }
  void SetNewId(spv::Id const old_id, spv::Id const new_id);

  /// ids from the new id space that have been claimed (faster than searching
  /// through new_id_)
  std::unordered_set<spv::Id> claimed_new_ids_;
  void ClaimNewId(spv::Id const new_id) { claimed_new_ids_.insert(new_id); }
  bool IsNewIdClaimed(spv::Id const new_id) const {
    return claimed_new_ids_.find(new_id) != claimed_new_ids_.end();
  }

  // queries for old ids
  bool IsOldIdUnmapped(spv::Id const old_id) const {
    return GetNewId(old_id) == unmapped_;
  }
  bool IsOldIdUnused(spv::Id const old_id) const {
    return GetNewId(old_id) == unused_;
  }

  // helper functions for printing ids (useful for debugging)
  std::string IdAsString(spv::Id const id) const;
  void PrintNewIds() const;
};

}  // namespace opt
}  // namespace spvtools
