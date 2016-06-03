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

#include <val/BasicBlock.h>
#include <val/Construct.h>
#include <val/Function.h>
#include <val/ValidationState.h>

using std::list;
using std::string;
using std::vector;

namespace libspirv {
namespace {

void printDot(const BasicBlock& other, const ValidationState_t& module) {
  string block_string;
  if (other.get_successors().empty()) {
    block_string += "end ";
  } else {
    for (auto& block : other.get_successors()) {
      block_string += module.getIdOrName(block->get_id()) + " ";
    }
  }
  printf("%10s -> {%s\b}\n", module.getIdOrName(other.get_id()).c_str(),
         block_string.c_str());
}
}

Function::Function(uint32_t id, uint32_t result_type_id,
                   SpvFunctionControlMask function_control,
                   uint32_t function_type_id, ValidationState_t& module)
    : module_(module),
      id_(id),
      function_type_id_(function_type_id),
      result_type_id_(result_type_id),
      function_control_(function_control),
      declaration_type_(FunctionDecl::kFunctionDeclUnknown),
      blocks_(),
      current_block_(nullptr),
      cfg_constructs_(),
      variable_ids_(),
      parameter_ids_() {}

bool Function::IsFirstBlock(uint32_t id) const {
  return !ordered_blocks_.empty() && *get_first_block() == id;
}

spv_result_t Function::RegisterFunctionParameter(uint32_t id,
                                                 uint32_t type_id) {
  assert(module_.in_function_body() == true &&
         "RegisterFunctionParameter can only be called when parsing the binary "
         "outside of another function");
  assert(get_current_block() == nullptr &&
         "RegisterFunctionParameter can only be called when parsing the binary "
         "ouside of a block");
  // TODO(umar): Validate function parameter type order and count
  // TODO(umar): Use these variables to validate parameter type
  (void)id;
  (void)type_id;
  return SPV_SUCCESS;
}

spv_result_t Function::RegisterLoopMerge(uint32_t merge_id,
                                         uint32_t continue_id) {
  RegisterBlock(merge_id, false);
  RegisterBlock(continue_id, false);
  cfg_constructs_.emplace_back(get_current_block(), &blocks_.at(merge_id),
                               &blocks_.at(continue_id));

  return SPV_SUCCESS;
}

spv_result_t Function::RegisterSelectionMerge(uint32_t merge_id) {
  RegisterBlock(merge_id, false);
  cfg_constructs_.emplace_back(get_current_block(), &blocks_.at(merge_id));
  return SPV_SUCCESS;
}

void Function::printDotGraph() const {
  if (get_first_block()) {
    string func_name(module_.getIdOrName(id_));
    printf("digraph %s {\n", func_name.c_str());
    printBlocks();
    printf("}\n");
  }
}

void Function::printBlocks() const {
  if (get_first_block()) {
    printf("%10s -> %s\n", module_.getIdOrName(id_).c_str(),
           module_.getIdOrName(get_first_block()->get_id()).c_str());
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

spv_result_t Function::RegisterBlock(uint32_t id, bool is_definition) {
  assert(module_.in_function_body() == true &&
         "RegisterBlocks can only be called when parsing a binary inside of a "
         "function");
  assert(module_.getLayoutSection() !=
             ModuleLayoutSection::kLayoutFunctionDeclarations &&
         "RegisterBlocks cannot be called within a function declaration");
  assert(
      declaration_type_ == FunctionDecl::kFunctionDeclDefinition &&
      "RegisterBlocks can only be called after declaration_type_ is defined");

  std::unordered_map<uint32_t, BasicBlock>::iterator inserted_block;
  bool success = false;
  tie(inserted_block, success) = blocks_.insert({id, BasicBlock(id)});
  if (is_definition) {  // new block definition
    assert(get_current_block() == nullptr &&
           "Register Block can only be called when parsing a binary outside of "
           "a BasicBlock");

    undefined_blocks_.erase(id);
    current_block_ = &inserted_block->second;
    ordered_blocks_.push_back(current_block_);
    if (IsFirstBlock(id)) current_block_->set_reachability(true);
  } else if (success) {  // Block doesn't exsist but this is not a definition
    undefined_blocks_.insert(id);
  }

  return SPV_SUCCESS;
}

void Function::RegisterBlockEnd(vector<uint32_t> next_list,
                                SpvOp branch_instruction) {
  assert(module_.in_function_body() == true &&
         "RegisterBlockEnd can only be called when parsing a binary in a "
         "function");
  assert(
      get_current_block() &&
      "RegisterBlockEnd can only be called when parsing a binary in a block");

  vector<BasicBlock*> next_blocks;
  next_blocks.reserve(next_list.size());

  std::unordered_map<uint32_t, BasicBlock>::iterator inserted_block;
  bool success;
  for (uint32_t id : next_list) {
    tie(inserted_block, success) = blocks_.insert({id, BasicBlock(id)});
    if (success) {
      undefined_blocks_.insert(id);
    }
    next_blocks.push_back(&inserted_block->second);
  }

  current_block_->RegisterBranchInstruction(branch_instruction);
  current_block_->RegisterSuccessors(next_blocks);
  current_block_ = nullptr;
  return;
}

size_t Function::get_block_count() const { return blocks_.size(); }

size_t Function::get_undefined_block_count() const {
  return undefined_blocks_.size();
}

const vector<BasicBlock*>& Function::get_blocks() const {
  return ordered_blocks_;
}
vector<BasicBlock*>& Function::get_blocks() { return ordered_blocks_; }

const BasicBlock* Function::get_current_block() const { return current_block_; }
BasicBlock* Function::get_current_block() { return current_block_; }

const list<Construct>& Function::get_constructs() const {
  return cfg_constructs_;
}
list<Construct>& Function::get_constructs() { return cfg_constructs_; }

const BasicBlock* Function::get_first_block() const {
  if (ordered_blocks_.empty()) return nullptr;
  return ordered_blocks_[0];
}
BasicBlock* Function::get_first_block() {
  if (ordered_blocks_.empty()) return nullptr;
  return ordered_blocks_[0];
}

bool Function::IsMergeBlock(uint32_t merge_block_id) const {
  const auto b = blocks_.find(merge_block_id);
  if (b != end(blocks_)) {
    return cfg_constructs_.end() !=
           find_if(begin(cfg_constructs_), end(cfg_constructs_),
                   [&](const Construct& construct) {
                     return construct.get_merge() == &b->second;
                   });
  } else {
    return false;
  }
}

}  /// namespace libspirv
