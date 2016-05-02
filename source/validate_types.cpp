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

#include <algorithm>
#include <cassert>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include "spirv/spirv.h"
#include "spirv_definition.h"
#include "validate.h"

using std::find;
using std::list;
using std::numeric_limits;
using std::string;
using std::unordered_set;
using std::vector;

using libspirv::kLayoutCapabilities;
using libspirv::kLayoutExtensions;
using libspirv::kLayoutExtInstImport;
using libspirv::kLayoutMemoryModel;
using libspirv::kLayoutEntryPoint;
using libspirv::kLayoutExecutionMode;
using libspirv::kLayoutDebug1;
using libspirv::kLayoutDebug2;
using libspirv::kLayoutAnnotations;
using libspirv::kLayoutTypes;
using libspirv::kLayoutFunctionDeclarations;
using libspirv::kLayoutFunctionDefinitions;
using libspirv::ModuleLayoutSection;

namespace {
bool IsInstructionInLayoutSection(ModuleLayoutSection layout, SpvOp op) {
  // See Section 2.4
  bool out = false;
  // clang-format off
  switch (layout) {
    case kLayoutCapabilities:  out = op == SpvOpCapability;    break;
    case kLayoutExtensions:    out = op == SpvOpExtension;     break;
    case kLayoutExtInstImport: out = op == SpvOpExtInstImport; break;
    case kLayoutMemoryModel:   out = op == SpvOpMemoryModel;   break;
    case kLayoutEntryPoint:    out = op == SpvOpEntryPoint;    break;
    case kLayoutExecutionMode: out = op == SpvOpExecutionMode; break;
    case kLayoutDebug1:
      switch (op) {
        case SpvOpSourceContinued:
        case SpvOpSource:
        case SpvOpSourceExtension:
        case SpvOpString:
          out = true;
          break;
        default: break;
      }
      break;
    case kLayoutDebug2:
      switch (op) {
        case SpvOpName:
        case SpvOpMemberName:
          out = true;
          break;
        default: break;
      }
      break;
    case kLayoutAnnotations:
      switch (op) {
        case SpvOpDecorate:
        case SpvOpMemberDecorate:
        case SpvOpGroupDecorate:
        case SpvOpGroupMemberDecorate:
        case SpvOpDecorationGroup:
          out = true;
          break;
        default: break;
      }
      break;
    case kLayoutTypes:
      switch (op) {
        case SpvOpTypeVoid:
        case SpvOpTypeBool:
        case SpvOpTypeInt:
        case SpvOpTypeFloat:
        case SpvOpTypeVector:
        case SpvOpTypeMatrix:
        case SpvOpTypeImage:
        case SpvOpTypeSampler:
        case SpvOpTypeSampledImage:
        case SpvOpTypeArray:
        case SpvOpTypeRuntimeArray:
        case SpvOpTypeStruct:
        case SpvOpTypeOpaque:
        case SpvOpTypePointer:
        case SpvOpTypeFunction:
        case SpvOpTypeEvent:
        case SpvOpTypeDeviceEvent:
        case SpvOpTypeReserveId:
        case SpvOpTypeQueue:
        case SpvOpTypePipe:
        case SpvOpTypeForwardPointer:
        case SpvOpConstantTrue:
        case SpvOpConstantFalse:
        case SpvOpConstant:
        case SpvOpConstantComposite:
        case SpvOpConstantSampler:
        case SpvOpConstantNull:
        case SpvOpSpecConstantTrue:
        case SpvOpSpecConstantFalse:
        case SpvOpSpecConstant:
        case SpvOpSpecConstantComposite:
        case SpvOpSpecConstantOp:
        case SpvOpVariable:
        case SpvOpLine:
        case SpvOpNoLine:
          out = true;
          break;
        default: break;
      }
      break;
    case kLayoutFunctionDeclarations:
    case kLayoutFunctionDefinitions:
      // NOTE: These instructions should NOT be in these layout sections
      switch (op) {
        case SpvOpCapability:
        case SpvOpExtension:
        case SpvOpExtInstImport:
        case SpvOpMemoryModel:
        case SpvOpEntryPoint:
        case SpvOpExecutionMode:
        case SpvOpSourceContinued:
        case SpvOpSource:
        case SpvOpSourceExtension:
        case SpvOpString:
        case SpvOpName:
        case SpvOpMemberName:
        case SpvOpDecorate:
        case SpvOpMemberDecorate:
        case SpvOpGroupDecorate:
        case SpvOpGroupMemberDecorate:
        case SpvOpDecorationGroup:
        case SpvOpTypeVoid:
        case SpvOpTypeBool:
        case SpvOpTypeInt:
        case SpvOpTypeFloat:
        case SpvOpTypeVector:
        case SpvOpTypeMatrix:
        case SpvOpTypeImage:
        case SpvOpTypeSampler:
        case SpvOpTypeSampledImage:
        case SpvOpTypeArray:
        case SpvOpTypeRuntimeArray:
        case SpvOpTypeStruct:
        case SpvOpTypeOpaque:
        case SpvOpTypePointer:
        case SpvOpTypeFunction:
        case SpvOpTypeEvent:
        case SpvOpTypeDeviceEvent:
        case SpvOpTypeReserveId:
        case SpvOpTypeQueue:
        case SpvOpTypePipe:
        case SpvOpTypeForwardPointer:
        case SpvOpConstantTrue:
        case SpvOpConstantFalse:
        case SpvOpConstant:
        case SpvOpConstantComposite:
        case SpvOpConstantSampler:
        case SpvOpConstantNull:
        case SpvOpSpecConstantTrue:
        case SpvOpSpecConstantFalse:
        case SpvOpSpecConstant:
        case SpvOpSpecConstantComposite:
        case SpvOpSpecConstantOp:
          out = false;
          break;
      default:
        out = true;
        break;
      }
  }
  // clang-format on
  return out;
}

}  // anonymous namespace

namespace libspirv {

void message(std::string file, size_t line, std::string name) {
  std::cout << file << ":" << line << ": " << name << std::endl;
}

ValidationState_t::ValidationState_t(spv_diagnostic* diagnostic,
                                     const spv_const_context context)
    : diagnostic_(diagnostic),
      instruction_counter_(0),
      unresolved_forward_ids_{},
      operand_names_{},
      current_layout_section_(kLayoutCapabilities),
      module_functions_(),
      module_capabilities_(0u),
      grammar_(context),
      addressing_model_(SpvAddressingModelLogical),
      memory_model_(SpvMemoryModelSimple),
      in_function_(false) {}

spv_result_t ValidationState_t::forwardDeclareId(uint32_t id) {
  unresolved_forward_ids_.insert(id);
  return SPV_SUCCESS;
}

spv_result_t ValidationState_t::removeIfForwardDeclared(uint32_t id) {
  unresolved_forward_ids_.erase(id);
  return SPV_SUCCESS;
}

void ValidationState_t::assignNameToId(uint32_t id, string name) {
  operand_names_[id] = name;
}

string ValidationState_t::getIdName(uint32_t id) const {
  std::stringstream out;
  out << id;
  if (operand_names_.find(id) != end(operand_names_)) {
    out << "[" << operand_names_.at(id) << "]";
  }
  return out.str();
}

string ValidationState_t::getIdOrName(uint32_t id) const {
  std::stringstream out;
  if (operand_names_.find(id) != end(operand_names_)) {
    out << operand_names_.at(id);
  } else {
    out << id;
  }
  return out.str();
}

size_t ValidationState_t::unresolvedForwardIdCount() const {
  return unresolved_forward_ids_.size();
}

vector<uint32_t> ValidationState_t::unresolvedForwardIds() const {
  vector<uint32_t> out(begin(unresolved_forward_ids_),
                       end(unresolved_forward_ids_));
  return out;
}

bool ValidationState_t::isDefinedId(uint32_t id) const {
  return usedefs_.FindDef(id).first;
}

// Increments the instruction count. Used for diagnostic
int ValidationState_t::incrementInstructionCount() {
  return instruction_counter_++;
}

ModuleLayoutSection ValidationState_t::getLayoutSection() const {
  return current_layout_section_;
}

void ValidationState_t::progressToNextLayoutSectionOrder() {
  // Guard against going past the last element(kLayoutFunctionDefinitions)
  if (current_layout_section_ <= kLayoutFunctionDefinitions) {
    current_layout_section_ =
        static_cast<ModuleLayoutSection>(current_layout_section_ + 1);
  }
}

bool ValidationState_t::isOpcodeInCurrentLayoutSection(SpvOp op) {
  return IsInstructionInLayoutSection(current_layout_section_, op);
}

DiagnosticStream ValidationState_t::diag(spv_result_t error_code) const {
  return libspirv::DiagnosticStream(
      {0, 0, static_cast<size_t>(instruction_counter_)}, diagnostic_,
      error_code);
}

list<Function>& ValidationState_t::get_functions() {
  return module_functions_;
}

Function& ValidationState_t::get_current_function() {
  assert(in_function_body());
  return module_functions_.back();
}

bool ValidationState_t::in_function_body() const { return in_function_; }

bool ValidationState_t::in_block() const {
  return module_functions_.back().in_block();
}

void ValidationState_t::RegisterCapability(SpvCapability cap) {
  module_capabilities_ |= SPV_CAPABILITY_AS_MASK(cap);
  spv_operand_desc desc;
  if (SPV_SUCCESS ==
      grammar_.lookupOperand(SPV_OPERAND_TYPE_CAPABILITY, cap, &desc))
    libspirv::ForEach(desc->capabilities,
                      [this](SpvCapability c) { RegisterCapability(c); });
}

bool ValidationState_t::hasCapability(SpvCapability cap) const {
  return (module_capabilities_ & SPV_CAPABILITY_AS_MASK(cap)) != 0;
}

bool ValidationState_t::HasAnyOf(spv_capability_mask_t capabilities) const {
  if (!capabilities)
    return true;  // No capabilities requested: trivially satisfied.
  bool found = false;
  libspirv::ForEach(capabilities, [&found, this](SpvCapability c) {
    found |= hasCapability(c);
  });
  return found;
}
	
void ValidationState_t::setAddressingModel(SpvAddressingModel am) {
  addressing_model_ = am;
}

SpvAddressingModel ValidationState_t::getAddressingModel() const {
  return addressing_model_;
}

void ValidationState_t::setMemoryModel(SpvMemoryModel mm) {
  memory_model_ = mm;
}

SpvMemoryModel ValidationState_t::getMemoryModel() const {
  return memory_model_;
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

bool Function::in_block() const { return static_cast<bool>(current_block_); }

bool Function::IsFirstBlock(uint32_t id) const {
  return *get_first_block() == id;
}

spv_result_t ValidationState_t::RegisterFunction(
    uint32_t id, uint32_t ret_type_id, SpvFunctionControlMask function_control,
    uint32_t function_type_id) {
  assert(in_function_ == false &&
         "Function instructions can not be declared in a function");
  assert(in_function_ == false &&
         "Function instructions can not be declared in a function");
  in_function_ = true;
  module_functions_.emplace_back(id, ret_type_id, function_control,
                                 function_type_id, *this);

  // TODO(umar): validate function type and type_id

  return SPV_SUCCESS;
}

spv_result_t ValidationState_t::RegisterFunctionEnd() {
  assert(in_function_body() == true &&
         "Function end can only be called in functions");
  assert(in_block() == false && "Function end cannot be called inside a block");
  in_function_ = false;
  return SPV_SUCCESS;
}

spv_result_t Function::RegisterFunctionParameter(uint32_t id,
                                                 uint32_t type_id) {
  assert(module_.in_function_body() == true &&
         "Function parameter instructions cannot be declared outside of a "
         "function");
  assert(in_block() == false &&
         "Function parameters cannot be called in blocks");
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
  cfg_constructs_.emplace_back(&get_current_block(), &blocks_.at(merge_id),
                               &blocks_.at(continue_id));

  return SPV_SUCCESS;
}

spv_result_t Function::RegisterSelectionMerge(uint32_t merge_id) {
  RegisterBlock(merge_id, false);
  cfg_constructs_.emplace_back(&get_current_block(), &blocks_.at(merge_id));
  return SPV_SUCCESS;
}

void printDot(const BasicBlock& other, const ValidationState_t& module) {
  string block_string;
  if (other.get_successors().empty()) {
    block_string += "end ";
  } else {
    for (auto& block : other.get_successors()) {
      block_string += module.getIdOrName(block->get_id()) + " ";
    }
  }
  printf("%10s -> {%s\b}\n", module.getIdOrName(other.get_id()).c_str(), block_string.c_str());
}

void Function::printDotGraph() const {
  using namespace std;
  if(get_first_block()) {
    string func_name(module_.getIdOrName(id_));
    printf("digraph %s {\n", func_name.c_str());
    printBlocks();
    printf("}\n");
  }
}

void Function::printBlocks() const {
  if(get_first_block()) {
    printf("%10s -> %s\n",
           module_.getIdOrName(id_).c_str() , module_.getIdOrName(get_first_block()->get_id()).c_str());
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
         "Blocks can only exsist in functions");
  assert(module_.getLayoutSection() !=
             ModuleLayoutSection::kLayoutFunctionDeclarations &&
         "Function declartions must appear before function definitions");
  assert(declaration_type_ == FunctionDecl::kFunctionDeclDefinition &&
         "Function declaration type should have already been defined");

  std::unordered_map<uint32_t, BasicBlock>::iterator tmp;
  bool success = false;
  tie(tmp, success) = blocks_.insert({id, BasicBlock(id)});
  if (is_definition) {  // new block definition
    assert(in_block() == false && "Blocks cannot be nested");

    undefined_blocks_.erase(id);
    current_block_ = &tmp->second;
    ordered_blocks_.push_back(current_block_);
  } else if (success) {  // Block doesn't exsist but this is not a definition
    undefined_blocks_.insert(id);
  }

  return SPV_SUCCESS;
}

spv_result_t Function::RegisterBlockEnd() {
  assert(module_.in_function_body() == true &&
         "Branch instruction can only be called in a function");
  assert(in_block() == true &&
         "Branch instruction can only be called in a block");
  current_block_ = nullptr;
  return SPV_SUCCESS;
}

spv_result_t Function::RegisterBlockEnd(uint32_t next_id) {
  assert(module_.in_function_body() == true &&
         "Branch instruction can only be called in a function");
  assert(in_block() == true &&
         "Branch instruction can only be called in a block");

  std::unordered_map<uint32_t, BasicBlock>::iterator tmp;
  bool success;
  tie(tmp, success) = blocks_.insert({next_id, BasicBlock(next_id)});
  if (success) {
    undefined_blocks_.insert(next_id);
  }
  current_block_->RegisterSuccessor(tmp->second);

  current_block_ = nullptr;
  return SPV_SUCCESS;
}

spv_result_t Function::RegisterBlockEnd(vector<uint32_t> next_list) {
  assert(module_.in_function_body() == true &&
         "Branch instruction can only be called in a function");
  assert(in_block() == true &&
         "Branch instruction can only be called in a block");

  vector<BasicBlock*> next_blocks;
  next_blocks.reserve(next_list.size());

  std::unordered_map<uint32_t, BasicBlock>::iterator tmp;
  bool success;
  for (uint32_t id : next_list) {
    tie(tmp, success) = blocks_.insert({id, BasicBlock(id)});
    if (success) {
      undefined_blocks_.insert(id);
    }
    next_blocks.push_back(&tmp->second);
  }

  current_block_->RegisterSuccessor(next_blocks);
  current_block_ = nullptr;
  return SPV_SUCCESS;
}

size_t Function::get_block_count() const { return blocks_.size(); }

size_t Function::get_undefined_block_count() const {
  return undefined_blocks_.size();
}

const vector<BasicBlock*>& Function::get_blocks() const {
  return ordered_blocks_;
}
vector<BasicBlock*>& Function::get_blocks() { return ordered_blocks_; }

const BasicBlock& Function::get_current_block() const {
  return *current_block_;
}
BasicBlock& Function::get_current_block() { return *current_block_; }

const list<CFConstruct>& Function::get_constructs() const {
  return cfg_constructs_;
}
list<CFConstruct>& Function::get_constructs() { return cfg_constructs_; }

const BasicBlock* Function::get_first_block() const {
  if (ordered_blocks_.empty()) return nullptr;
  return ordered_blocks_[0];
}
BasicBlock* Function::get_first_block() {
  if (ordered_blocks_.empty()) return nullptr;
  return ordered_blocks_[0];
}

BasicBlock::BasicBlock(uint32_t id)
    : id_(id),
      immediate_dominator_(nullptr),
      predecessors_(),
      successors_() {}

void BasicBlock::SetImmediateDominator(BasicBlock* dom_block) {
  immediate_dominator_ = dom_block;
}

const BasicBlock* BasicBlock::GetImmediateDominator() const {
  return immediate_dominator_;
}

BasicBlock* BasicBlock::GetImmediateDominator() { return immediate_dominator_; }

void BasicBlock::RegisterSuccessor(BasicBlock& next) {
  next.predecessors_.push_back(this);
  successors_.push_back(&next);
}

void BasicBlock::RegisterSuccessor(vector<BasicBlock*> next_blocks) {
  for (auto& block : next_blocks) {
    block->predecessors_.push_back(this);
    successors_.push_back(block);
  }
}

bool Function::IsMergeBlock(uint32_t merge_block_id) const {
  const auto b = blocks_.find(merge_block_id);
  if (b != end(blocks_)) {
    return cfg_constructs_.end() !=
           find_if(begin(cfg_constructs_), end(cfg_constructs_),
                   [&](const CFConstruct& construct) {
                     return construct.merge_block_ == &b->second;
                   });
  } else {
    return false;
  }
}

BasicBlock::DominatorIterator::DominatorIterator() : current_(nullptr) {}
BasicBlock::DominatorIterator::DominatorIterator(BasicBlock* block)
    : current_(block) {}

BasicBlock::DominatorIterator& BasicBlock::DominatorIterator::operator++() {
  if (current_ == current_->GetImmediateDominator()) {
    current_ = nullptr;
  } else {
    current_ = current_->GetImmediateDominator();
  }
  return *this;
}

BasicBlock::DominatorIterator BasicBlock::dom_begin() {
  return DominatorIterator(this);
}

BasicBlock::DominatorIterator BasicBlock::dom_end() {
  return DominatorIterator();
}

bool operator==(const BasicBlock::DominatorIterator& lhs,
                const BasicBlock::DominatorIterator& rhs) {
  return lhs.current_ == rhs.current_;
}

bool operator!=(const BasicBlock::DominatorIterator& lhs,
                const BasicBlock::DominatorIterator& rhs) {
  return !(lhs == rhs);
}

BasicBlock*& BasicBlock::DominatorIterator::operator*() { return current_; }
}
