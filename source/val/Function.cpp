// Copyright (c) 2015-2016 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

#include "val/Function.h"

#include <cassert>

#include <algorithm>
#include <utility>

#include "val/BasicBlock.h"
#include "val/Construct.h"
#include "val/ValidationState.h"

using std::ignore;
using std::list;
using std::make_pair;
using std::pair;
using std::string;
using std::tie;
using std::vector;

namespace libspirv {
namespace {

void printDot(const BasicBlock& other, const ValidationState_t& module) {
  string block_string;
  if (other.successors()->empty()) {
    block_string += "end ";
  } else {
    for (auto block : *other.successors()) {
      block_string += module.getIdOrName(block->id()) + " ";
    }
  }
  printf("%10s -> {%s\b}\n", module.getIdOrName(other.id()).c_str(),
         block_string.c_str());
}
}  /// namespace

Function::Function(uint32_t function_id, uint32_t result_type_id,
                   SpvFunctionControlMask function_control,
                   uint32_t function_type_id, ValidationState_t& module)
    : module_(module),
      id_(function_id),
      function_type_id_(function_type_id),
      result_type_id_(result_type_id),
      function_control_(function_control),
      declaration_type_(FunctionDecl::kFunctionDeclUnknown),
      end_has_been_registered_(false),
      blocks_(),
      current_block_(nullptr),
      pseudo_entry_block_(0),
      pseudo_exit_block_(kInvalidId),
      pseudo_entry_blocks_({&pseudo_entry_block_}),
      pseudo_exit_blocks_({&pseudo_exit_block_}),
      cfg_constructs_(),
      variable_ids_(),
      parameter_ids_() {}

bool Function::IsFirstBlock(uint32_t block_id) const {
  return !ordered_blocks_.empty() && *first_block() == block_id;
}

spv_result_t Function::RegisterFunctionParameter(uint32_t parameter_id,
                                                 uint32_t type_id) {
  assert(module_.in_function_body() == true &&
         "RegisterFunctionParameter can only be called when parsing the binary "
         "outside of another function");
  assert(current_block_ == nullptr &&
         "RegisterFunctionParameter can only be called when parsing the binary "
         "ouside of a block");
  // TODO(umar): Validate function parameter type order and count
  // TODO(umar): Use these variables to validate parameter type
  (void)parameter_id;
  (void)type_id;
  return SPV_SUCCESS;
}

spv_result_t Function::RegisterLoopMerge(uint32_t merge_id,
                                         uint32_t continue_id) {
  RegisterBlock(merge_id, false);
  RegisterBlock(continue_id, false);
  BasicBlock& merge_block = blocks_.at(merge_id);
  BasicBlock& continue_block = blocks_.at(continue_id);
  assert(current_block_ &&
         "RegisterLoopMerge must be called when called within a block");

  current_block_->set_type(kBlockTypeLoop);
  merge_block.set_type(kBlockTypeMerge);
  continue_block.set_type(kBlockTypeContinue);
  cfg_constructs_.emplace_back(ConstructType::kLoop, current_block_,
                               &merge_block);
  Construct& loop_construct = cfg_constructs_.back();
  cfg_constructs_.emplace_back(ConstructType::kContinue, &continue_block);
  Construct& continue_construct = cfg_constructs_.back();
  continue_construct.set_corresponding_constructs({&loop_construct});
  loop_construct.set_corresponding_constructs({&continue_construct});

  return SPV_SUCCESS;
}

spv_result_t Function::RegisterSelectionMerge(uint32_t merge_id) {
  RegisterBlock(merge_id, false);
  BasicBlock& merge_block = blocks_.at(merge_id);
  current_block_->set_type(kBlockTypeHeader);
  merge_block.set_type(kBlockTypeMerge);

  cfg_constructs_.emplace_back(ConstructType::kSelection, current_block(),
                               &merge_block);
  return SPV_SUCCESS;
}

void Function::PrintDotGraph() const {
  if (first_block()) {
    string func_name(module_.getIdOrName(id_));
    printf("digraph %s {\n", func_name.c_str());
    PrintBlocks();
    printf("}\n");
  }
}

void Function::PrintBlocks() const {
  if (first_block()) {
    printf("%10s -> %s\n", module_.getIdOrName(id_).c_str(),
           module_.getIdOrName(first_block()->id()).c_str());
    for (const auto& block : blocks_) {
      printDot(block.second, module_);
    }
  }
}

spv_result_t Function::RegisterSetFunctionDeclType(FunctionDecl type) {
  assert(declaration_type_ == FunctionDecl::kFunctionDeclUnknown);
  declaration_type_ = type;
  return SPV_SUCCESS;
}

spv_result_t Function::RegisterBlock(uint32_t block_id, bool is_definition) {
  assert(module_.in_function_body() == true &&
         "RegisterBlocks can only be called when parsing a binary inside of a "
         "function");
  assert(module_.current_layout_section() !=
             ModuleLayoutSection::kLayoutFunctionDeclarations &&
         "RegisterBlocks cannot be called within a function declaration");
  assert(
      declaration_type_ == FunctionDecl::kFunctionDeclDefinition &&
      "RegisterBlocks can only be called after declaration_type_ is defined");

  std::unordered_map<uint32_t, BasicBlock>::iterator inserted_block;
  bool success = false;
  tie(inserted_block, success) =
      blocks_.insert({block_id, BasicBlock(block_id)});
  if (is_definition) {  // new block definition
    assert(current_block_ == nullptr &&
           "Register Block can only be called when parsing a binary outside of "
           "a BasicBlock");

    undefined_blocks_.erase(block_id);
    current_block_ = &inserted_block->second;
    ordered_blocks_.push_back(current_block_);
    if (IsFirstBlock(block_id)) current_block_->set_reachable(true);
  } else if (success) {  // Block doesn't exsist but this is not a definition
    undefined_blocks_.insert(block_id);
  }

  return SPV_SUCCESS;
}

void Function::RegisterBlockEnd(vector<uint32_t> next_list,
                                SpvOp branch_instruction) {
  assert(module_.in_function_body() == true &&
         "RegisterBlockEnd can only be called when parsing a binary in a "
         "function");
  assert(
      current_block_ &&
      "RegisterBlockEnd can only be called when parsing a binary in a block");

  vector<BasicBlock*> next_blocks;
  next_blocks.reserve(next_list.size());

  std::unordered_map<uint32_t, BasicBlock>::iterator inserted_block;
  bool success;
  for (uint32_t successor_id : next_list) {
    tie(inserted_block, success) =
        blocks_.insert({successor_id, BasicBlock(successor_id)});
    if (success) {
      undefined_blocks_.insert(successor_id);
    }
    next_blocks.push_back(&inserted_block->second);
  }

  current_block_->RegisterBranchInstruction(branch_instruction);
  current_block_->RegisterSuccessors(next_blocks);
  current_block_ = nullptr;
  return;
}

void Function::RegisterFunctionEnd() {
  if (!end_has_been_registered_) {
    end_has_been_registered_ = true;

    // Compute the successors of the pseudo-entry block, and
    // the predecessors of the pseudo exit block.
    vector<BasicBlock*> sources;
    vector<BasicBlock*> sinks;
    for (const auto b : ordered_blocks_) {
      if (b->predecessors()->empty()) sources.push_back(b);
      if (b->successors()->empty()) sinks.push_back(b);
    }
    pseudo_entry_block_.SetSuccessorsUnsafe(std::move(sources));
    pseudo_exit_block_.SetPredecessorsUnsafe(std::move(sinks));
  }
}

size_t Function::block_count() const { return blocks_.size(); }

size_t Function::undefined_block_count() const {
  return undefined_blocks_.size();
}

const vector<BasicBlock*>& Function::ordered_blocks() const {
  return ordered_blocks_;
}
vector<BasicBlock*>& Function::ordered_blocks() { return ordered_blocks_; }

const BasicBlock* Function::current_block() const { return current_block_; }
BasicBlock* Function::current_block() { return current_block_; }

BasicBlock* Function::pseudo_entry_block() { return &pseudo_entry_block_; }
const BasicBlock* Function::pseudo_entry_block() const {
  return &pseudo_entry_block_;
}

BasicBlock* Function::pseudo_exit_block() { return &pseudo_exit_block_; }
const BasicBlock* Function::pseudo_exit_block() const {
  return &pseudo_exit_block_;
}

const list<Construct>& Function::constructs() const { return cfg_constructs_; }
list<Construct>& Function::constructs() { return cfg_constructs_; }

const BasicBlock* Function::first_block() const {
  if (ordered_blocks_.empty()) return nullptr;
  return ordered_blocks_[0];
}
BasicBlock* Function::first_block() {
  if (ordered_blocks_.empty()) return nullptr;
  return ordered_blocks_[0];
}

bool Function::IsBlockType(uint32_t merge_block_id, BlockType type) const {
  bool ret = false;
  const BasicBlock* block;
  tie(block, ignore) = GetBlock(merge_block_id);
  if (block) {
    ret = block->is_type(type);
  }
  return ret;
}

pair<const BasicBlock*, bool> Function::GetBlock(uint32_t block_id) const {
  const auto b = blocks_.find(block_id);
  if (b != end(blocks_)) {
    const BasicBlock* block = &(b->second);
    bool defined =
        undefined_blocks_.find(block->id()) == end(undefined_blocks_);
    return make_pair(block, defined);
  } else {
    return make_pair(nullptr, false);
  }
}

pair<BasicBlock*, bool> Function::GetBlock(uint32_t block_id) {
  const BasicBlock* out;
  bool defined;
  tie(out, defined) = const_cast<const Function*>(this)->GetBlock(block_id);
  return make_pair(const_cast<BasicBlock*>(out), defined);
}
}  /// namespace libspirv
