// Copyright (c) 2017 Pierre Moreau
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

#include "spirv-tools/linker.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/assembly_grammar.h"
#include "source/diagnostic.h"
#include "source/opt/build_module.h"
#include "source/opt/compact_ids_pass.h"
#include "source/opt/decoration_manager.h"
#include "source/opt/ir_loader.h"
#include "source/opt/pass_manager.h"
#include "source/opt/remove_duplicates_pass.h"
#include "source/opt/type_manager.h"
#include "source/spirv_constant.h"
#include "source/spirv_target_env.h"
#include "source/util/make_unique.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace {

using opt::Instruction;
using opt::IRContext;
using opt::Module;
using opt::PassManager;
using opt::RemoveDuplicatesPass;
using opt::analysis::DecorationManager;
using opt::analysis::DefUseManager;
using opt::analysis::Type;
using opt::analysis::TypeManager;

// Stores various information about an imported or exported symbol.
struct LinkageSymbolInfo {
  SpvId id;          // ID of the symbol
  SpvId type_id;     // ID of the type of the symbol
  std::string name;  // unique name defining the symbol and used for matching
                     // imports and exports together
  std::vector<SpvId> parameter_ids;  // ID of the parameters of the symbol, if
                                     // it is a function
};
struct LinkageEntry {
  LinkageSymbolInfo imported_symbol;
  LinkageSymbolInfo exported_symbol;

  LinkageEntry(const LinkageSymbolInfo& import_info,
               const LinkageSymbolInfo& export_info)
      : imported_symbol(import_info), exported_symbol(export_info) {}
};
using LinkageTable = std::vector<LinkageEntry>;

// Shifts the IDs used in each binary of |modules| so that they occupy a
// disjoint range from the other binaries, and compute the new ID bound which
// is returned in |max_id_bound|.
//
// Both |modules| and |max_id_bound| should not be null, and |modules| should
// not be empty either. Furthermore |modules| should not contain any null
// pointers.
spv_result_t ShiftIdsInModules(const MessageConsumer& consumer,
                               std::vector<opt::Module*>* modules,
                               uint32_t* max_id_bound);

// Consolidates eventual constant WorkgroupSize built-in into LocalSize
// execution mode.
//
// Multiple constant WorkgroupSize into a single module seem to confuse
// some drivers.
//
// Neither |module| nor |context| should be a null pointer.
// TODO(pierremoreau): WorkgroupSize built-ins applied by a group decoration
//                     are currently not handled. (You could have a group being
//                     applied to a single ID.)
spv_result_t ConsolidateWorkgroupSize(const MessageConsumer& consumer,
                                      uint32_t moduleIndex, opt::Module* module,
                                      opt::IRContext* context);

// Generates the header for the linked module and returns it in |header|.
//
// |header| should not be null, |modules| should not be empty and pointers
// should be non-null. |max_id_bound| should be strictly greater than 0.
//
// TODO(pierremoreau): What to do when binaries use different versions of
//                     SPIR-V? For now, use the max of all versions found in
//                     the input modules.
spv_result_t GenerateHeader(const MessageConsumer& consumer,
                            const std::vector<opt::Module*>& modules,
                            uint32_t max_id_bound, opt::ModuleHeader* header);

// Merge all the modules from |in_modules| into a single module owned by
// |linked_context|.
//
// |linked_context| should not be null.
spv_result_t MergeModules(const MessageConsumer& consumer,
                          const std::vector<Module*>& in_modules,
                          const AssemblyGrammar& grammar,
                          IRContext* linked_context);

// Compute all pairs of import and export and return it in |linkings_to_do|.
//
// |linkings_to_do should not be null. Built-in symbols will be ignored.
//
// TODO(pierremoreau): Linkage attributes applied by a group decoration are
//                     currently not handled. (You could have a group being
//                     applied to a single ID.)
// TODO(pierremoreau): What should be the proper behaviour with built-in
//                     symbols?
spv_result_t GetImportExportPairs(const MessageConsumer& consumer,
                                  const opt::IRContext& linked_context,
                                  const DefUseManager& def_use_manager,
                                  const DecorationManager& decoration_manager,
                                  bool allow_partial_linkage,
                                  LinkageTable* linkings_to_do);

// Checks that for each pair of import and export, the import and export have
// the same type as well as the same decorations.
//
// TODO(pierremoreau): Decorations on functions parameters are currently not
// checked.
spv_result_t CheckImportExportCompatibility(const MessageConsumer& consumer,
                                            const LinkageTable& linkings_to_do,
                                            opt::IRContext* context);

// Remove linkage specific instructions, such as prototypes of imported
// functions, declarations of imported variables, import (and export if
// necessary) linkage attribtes.
//
// |linked_context| and |decoration_manager| should not be null, and the
// 'RemoveDuplicatePass' should be run first.
//
// TODO(pierremoreau): Linkage attributes applied by a group decoration are
//                     currently not handled. (You could have a group being
//                     applied to a single ID.)
spv_result_t RemoveLinkageSpecificInstructions(
    const MessageConsumer& consumer, const LinkerOptions& options,
    const LinkageTable& linkings_to_do, DecorationManager* decoration_manager,
    opt::IRContext* linked_context);

// Verify that the unique ids of each instruction in |linked_context| (i.e. the
// merged module) are truly unique. Does not check the validity of other ids
spv_result_t VerifyIds(const MessageConsumer& consumer,
                       opt::IRContext* linked_context);

spv_result_t ShiftIdsInModules(const MessageConsumer& consumer,
                               std::vector<opt::Module*>* modules,
                               uint32_t* max_id_bound) {
  spv_position_t position = {};

  if (modules == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|modules| of ShiftIdsInModules should not be null.";
  if (modules->empty())
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|modules| of ShiftIdsInModules should not be empty.";
  if (max_id_bound == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|max_id_bound| of ShiftIdsInModules should not be null.";

  uint32_t id_bound = modules->front()->IdBound() - 1u;
  for (auto module_iter = modules->begin() + 1; module_iter != modules->end();
       ++module_iter) {
    Module* module = *module_iter;
    module->ForEachInst([&id_bound](Instruction* insn) {
      insn->ForEachId([&id_bound](uint32_t* id) { *id += id_bound; });
    });
    id_bound += module->IdBound() - 1u;
    if (id_bound > 0x3FFFFF)
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_ID)
             << "The limit of IDs, 4194303, was exceeded:"
             << " " << id_bound << " is the current ID bound.";

    // Invalidate the DefUseManager
    module->context()->InvalidateAnalyses(opt::IRContext::kAnalysisDefUse);
  }
  ++id_bound;
  if (id_bound > 0x3FFFFF)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_ID)
           << "The limit of IDs, 4194303, was exceeded:"
           << " " << id_bound << " is the current ID bound.";

  *max_id_bound = id_bound;

  return SPV_SUCCESS;
}

spv_result_t ConsolidateWorkgroupSize(const MessageConsumer& consumer,
                                      std::uint32_t moduleIndex,
                                      opt::Module* module,
                                      opt::IRContext* context) {
  spv_position_t position = {};

  if (module == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "Input module " << moduleIndex << ": "
           << "|module| of ConsolidateWorkgroupSize should not be null.";
  if (context == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "Input module " << moduleIndex << ": "
           << "|context| of ConsolidateWorkgroupSize should not be null.";

  std::vector<SpvId> entry_point_ids;
  for (const auto& entry_point : module->entry_points()) {
    const SpvExecutionModel execution_model =
        static_cast<SpvExecutionModel>(entry_point.GetSingleWordInOperand(0u));
    if (execution_model != SpvExecutionModelGLCompute &&
        execution_model != SpvExecutionModelKernel)
      continue;

    entry_point_ids.push_back(entry_point.GetSingleWordInOperand(1u));
  }

  struct WorkgroupSizeData {
    Instruction* decoration;
    SpvId constant_id;
    uint32_t workgroup_size_x;
    uint32_t workgroup_size_y;
    uint32_t workgroup_size_z;
  };
  std::vector<WorkgroupSizeData> workgroup_sizes;

  opt::analysis::ConstantManager& constant_manager =
      *context->get_constant_mgr();
  for (auto& decoration : context->annotations()) {
    if (decoration.opcode() != SpvOpDecorate ||
        decoration.GetSingleWordInOperand(1u) != SpvDecorationBuiltIn ||
        decoration.GetSingleWordInOperand(2u) != SpvBuiltInWorkgroupSize)
      continue;

    const SpvId id = decoration.GetSingleWordInOperand(0u);
    const auto* constant = constant_manager.FindDeclaredConstant(id);
    // We only consolidate constants.
    if (constant == nullptr) continue;

    const auto* vec_constant = constant->AsVectorConstant();
    assert(vec_constant != nullptr);
    const auto components = vec_constant->GetComponents();
    if (components.size() != 3)
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
             << "Input module " << moduleIndex << ": "
             << "WorkgroupSize constants should be 3-component vectors; only "
             << components.size() << " components were retrieved.";

    workgroup_sizes.push_back({&decoration, id, components[0]->GetU32(),
                               components[1]->GetU32(),
                               components[2]->GetU32()});
  }

  if (workgroup_sizes.empty()) return SPV_SUCCESS;
  if (entry_point_ids.empty())
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "Input module " << moduleIndex << ": "
           << "No GLCompute or Kernel entry points found to match the "
              "different WorkgroupSize constants to.";
  if (entry_point_ids.size() > 1)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "Input module " << moduleIndex << ": "
           << "More than one GLCompute or Kernel entry point was found: unable "
              "to consolidate the WorkgroupSize constant.";
  if (workgroup_sizes.size() > 1)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "Input module " << moduleIndex << ": "
           << "More than one WorkgroupSize constant was found: they will not "
              "be consolidated into LocalSize execution mode.";

  const auto& workgroup_size = workgroup_sizes.front();
  const SpvId entry_point_id = entry_point_ids.front();
  bool has_matching_execution_mode = false;
  for (const auto& execution_mode : module->execution_modes()) {
    if (execution_mode.GetSingleWordInOperand(0u) != entry_point_id ||
        execution_mode.GetSingleWordInOperand(1u) != SpvExecutionModeLocalSize)
      continue;

    const uint32_t local_size_x = execution_mode.GetSingleWordInOperand(2u);
    const uint32_t local_size_y = execution_mode.GetSingleWordInOperand(3u);
    const uint32_t local_size_z = execution_mode.GetSingleWordInOperand(4u);
    if (local_size_x != workgroup_size.workgroup_size_x ||
        local_size_y != workgroup_size.workgroup_size_y ||
        local_size_z != workgroup_size.workgroup_size_z)
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
             << "Input module " << moduleIndex << ": "
             << "Unable to consolidate WorkgroupSize constant %"
             << workgroup_size.constant_id << ": entry point %"
             << entry_point_id << " already has a specified LocalSize, ("
             << local_size_x << "," << local_size_y << "," << local_size_z
             << "), which differs from the specified constant WorkgroupSize, ("
             << workgroup_size.workgroup_size_x << ","
             << workgroup_size.workgroup_size_y << ","
             << workgroup_size.workgroup_size_z << ").";

    // If we reach here, we have found an OpExecutionMode LocalSize for our
    // targeted entry point and it has the same size as what was specified by
    // the WorkgroupSize constant.
    has_matching_execution_mode = true;
    break;
  }

  if (!has_matching_execution_mode)
    module->AddExecutionMode(std::unique_ptr<Instruction>(new Instruction(
        context, SpvOpExecutionMode, 0u, 0u,
        {{SPV_OPERAND_TYPE_ID, {entry_point_id}},
         {SPV_OPERAND_TYPE_EXECUTION_MODE, {SpvExecutionModeLocalSize}},
         {SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER,
          {workgroup_size.workgroup_size_x}},
         {SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER,
          {workgroup_size.workgroup_size_y}},
         {SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER,
          {workgroup_size.workgroup_size_z}}})));

  context->KillInst(workgroup_size.decoration);

  return SPV_SUCCESS;
}

spv_result_t GenerateHeader(const MessageConsumer& consumer,
                            const std::vector<opt::Module*>& modules,
                            uint32_t max_id_bound, opt::ModuleHeader* header) {
  spv_position_t position = {};

  if (modules.empty())
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|modules| of GenerateHeader should not be empty.";
  if (max_id_bound == 0u)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|max_id_bound| of GenerateHeader should not be null.";

  uint32_t version = 0u;
  for (const auto& module : modules)
    version = std::max(version, module->version());

  header->magic_number = SpvMagicNumber;
  header->version = version;
  header->generator = SPV_GENERATOR_WORD(SPV_GENERATOR_KHRONOS_LINKER, 0);
  header->bound = max_id_bound;
  header->reserved = 0u;

  return SPV_SUCCESS;
}

spv_result_t MergeModules(const MessageConsumer& consumer,
                          const std::vector<Module*>& input_modules,
                          const AssemblyGrammar& grammar,
                          IRContext* linked_context) {
  spv_position_t position = {};

  if (linked_context == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|linked_module| of MergeModules should not be null.";
  Module* linked_module = linked_context->module();

  if (input_modules.empty()) return SPV_SUCCESS;

  for (const auto& module : input_modules)
    for (const auto& inst : module->capabilities())
      linked_module->AddCapability(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  for (const auto& module : input_modules)
    for (const auto& inst : module->extensions())
      linked_module->AddExtension(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  for (const auto& module : input_modules)
    for (const auto& inst : module->ext_inst_imports())
      linked_module->AddExtInstImport(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  do {
    const Instruction* memory_model_inst = input_modules[0]->GetMemoryModel();
    if (memory_model_inst == nullptr) break;

    uint32_t addressing_model = memory_model_inst->GetSingleWordOperand(0u);
    uint32_t memory_model = memory_model_inst->GetSingleWordOperand(1u);
    for (const auto& module : input_modules) {
      memory_model_inst = module->GetMemoryModel();
      if (memory_model_inst == nullptr) continue;

      if (addressing_model != memory_model_inst->GetSingleWordOperand(0u)) {
        spv_operand_desc initial_desc = nullptr, current_desc = nullptr;
        grammar.lookupOperand(SPV_OPERAND_TYPE_ADDRESSING_MODEL,
                              addressing_model, &initial_desc);
        grammar.lookupOperand(SPV_OPERAND_TYPE_ADDRESSING_MODEL,
                              memory_model_inst->GetSingleWordOperand(0u),
                              &current_desc);
        return DiagnosticStream(position, consumer, "", SPV_ERROR_INTERNAL)
               << "Conflicting addressing models: " << initial_desc->name
               << " vs " << current_desc->name << ".";
      }
      if (memory_model != memory_model_inst->GetSingleWordOperand(1u)) {
        spv_operand_desc initial_desc = nullptr, current_desc = nullptr;
        grammar.lookupOperand(SPV_OPERAND_TYPE_MEMORY_MODEL, memory_model,
                              &initial_desc);
        grammar.lookupOperand(SPV_OPERAND_TYPE_MEMORY_MODEL,
                              memory_model_inst->GetSingleWordOperand(1u),
                              &current_desc);
        return DiagnosticStream(position, consumer, "", SPV_ERROR_INTERNAL)
               << "Conflicting memory models: " << initial_desc->name << " vs "
               << current_desc->name << ".";
      }
    }

    if (memory_model_inst != nullptr)
      linked_module->SetMemoryModel(std::unique_ptr<Instruction>(
          memory_model_inst->Clone(linked_context)));
  } while (false);

  std::vector<std::pair<uint32_t, const char*>> entry_points;
  for (const auto& module : input_modules)
    for (const auto& inst : module->entry_points()) {
      const uint32_t model = inst.GetSingleWordInOperand(0);
      const char* const name =
          reinterpret_cast<const char*>(inst.GetInOperand(2).words.data());
      const auto i = std::find_if(
          entry_points.begin(), entry_points.end(),
          [model, name](const std::pair<uint32_t, const char*>& v) {
            return v.first == model && strcmp(name, v.second) == 0;
          });
      if (i != entry_points.end()) {
        spv_operand_desc desc = nullptr;
        grammar.lookupOperand(SPV_OPERAND_TYPE_EXECUTION_MODEL, model, &desc);
        return DiagnosticStream(position, consumer, "", SPV_ERROR_INTERNAL)
               << "The entry point \"" << name << "\", with execution model "
               << desc->name << ", was already defined.";
      }
      linked_module->AddEntryPoint(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));
      entry_points.emplace_back(model, name);
    }

  for (const auto& module : input_modules)
    for (const auto& inst : module->execution_modes())
      linked_module->AddExecutionMode(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  for (const auto& module : input_modules)
    for (const auto& inst : module->debugs1())
      linked_module->AddDebug1Inst(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  for (const auto& module : input_modules)
    for (const auto& inst : module->debugs2())
      linked_module->AddDebug2Inst(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  for (const auto& module : input_modules)
    for (const auto& inst : module->debugs3())
      linked_module->AddDebug3Inst(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  for (const auto& module : input_modules)
    for (const auto& inst : module->ext_inst_debuginfo())
      linked_module->AddExtInstDebugInfo(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  // If the generated module uses SPIR-V 1.1 or higher, add an
  // OpModuleProcessed instruction about the linking step.
  if (linked_module->version() >= 0x10100) {
    const std::string processed_string("Linked by SPIR-V Tools Linker");
    const auto num_chars = processed_string.size();
    // Compute num words, accommodate the terminating null character.
    const auto num_words = (num_chars + 1 + 3) / 4;
    std::vector<uint32_t> processed_words(num_words, 0u);
    std::memcpy(processed_words.data(), processed_string.data(), num_chars);
    linked_module->AddDebug3Inst(std::unique_ptr<Instruction>(
        new Instruction(linked_context, SpvOpModuleProcessed, 0u, 0u,
                        {{SPV_OPERAND_TYPE_LITERAL_STRING, processed_words}})));
  }

  for (const auto& module : input_modules)
    for (const auto& inst : module->annotations())
      linked_module->AddAnnotationInst(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  // TODO(pierremoreau): Since the modules have not been validate, should we
  //                     expect SpvStorageClassFunction variables outside
  //                     functions?
  uint32_t num_global_values = 0u;
  for (const auto& module : input_modules) {
    for (const auto& inst : module->types_values()) {
      linked_module->AddType(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));
      num_global_values += inst.opcode() == SpvOpVariable;
    }
  }
  if (num_global_values > 0xFFFF)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INTERNAL)
           << "The limit of global values, 65535, was exceeded;"
           << " " << num_global_values << " global values were found.";

  // Process functions and their basic blocks
  for (const auto& module : input_modules) {
    for (const auto& func : *module) {
      std::unique_ptr<opt::Function> cloned_func(func.Clone(linked_context));
      linked_module->AddFunction(std::move(cloned_func));
    }
  }

  return SPV_SUCCESS;
}

spv_result_t GetImportExportPairs(const MessageConsumer& consumer,
                                  const opt::IRContext& linked_context,
                                  const DefUseManager& def_use_manager,
                                  const DecorationManager& decoration_manager,
                                  bool allow_partial_linkage,
                                  LinkageTable* linkings_to_do) {
  spv_position_t position = {};

  if (linkings_to_do == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|linkings_to_do| of GetImportExportPairs should not be empty.";

  std::vector<LinkageSymbolInfo> imports;
  std::unordered_map<std::string, std::vector<LinkageSymbolInfo>> exports;

  // Figure out the imports and exports
  for (const auto& decoration : linked_context.annotations()) {
    if (decoration.opcode() != SpvOpDecorate ||
        decoration.GetSingleWordInOperand(1u) != SpvDecorationLinkageAttributes)
      continue;

    const SpvId id = decoration.GetSingleWordInOperand(0u);
    // Ignore if the targeted symbol is a built-in
    bool is_built_in = false;
    for (const auto& id_decoration :
         decoration_manager.GetDecorationsFor(id, false)) {
      if (id_decoration->GetSingleWordInOperand(1u) == SpvDecorationBuiltIn) {
        is_built_in = true;
        break;
      }
    }
    if (is_built_in) {
      continue;
    }

    const uint32_t type = decoration.GetSingleWordInOperand(3u);

    LinkageSymbolInfo symbol_info;
    symbol_info.name =
        reinterpret_cast<const char*>(decoration.GetInOperand(2u).words.data());
    symbol_info.id = id;
    symbol_info.type_id = 0u;

    // Retrieve the type of the current symbol. This information will be used
    // when checking that the imported and exported symbols have the same
    // types.
    const Instruction* def_inst = def_use_manager.GetDef(id);
    if (def_inst == nullptr)
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "ID " << id << " is never defined:\n";

    if (def_inst->opcode() == SpvOpVariable) {
      symbol_info.type_id = def_inst->type_id();
    } else if (def_inst->opcode() == SpvOpFunction) {
      symbol_info.type_id = def_inst->GetSingleWordInOperand(1u);

      // range-based for loop calls begin()/end(), but never cbegin()/cend(),
      // which will not work here.
      for (auto func_iter = linked_context.module()->cbegin();
           func_iter != linked_context.module()->cend(); ++func_iter) {
        if (func_iter->result_id() != id) continue;
        func_iter->ForEachParam([&symbol_info](const Instruction* inst) {
          symbol_info.parameter_ids.push_back(inst->result_id());
        });
      }
    } else {
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Only global variables and functions can be decorated using"
             << " LinkageAttributes; " << id << " is neither of them.\n";
    }

    if (type == SpvLinkageTypeImport)
      imports.push_back(symbol_info);
    else if (type == SpvLinkageTypeExport)
      exports[symbol_info.name].push_back(symbol_info);
  }

  // Find the import/export pairs
  for (const auto& import : imports) {
    std::vector<LinkageSymbolInfo> possible_exports;
    const auto& exp = exports.find(import.name);
    if (exp != exports.end()) possible_exports = exp->second;
    if (possible_exports.empty() && !allow_partial_linkage)
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Unresolved external reference to \"" << import.name << "\".";
    else if (possible_exports.size() > 1u)
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Too many external references, " << possible_exports.size()
             << ", were found for \"" << import.name << "\".";

    if (!possible_exports.empty())
      linkings_to_do->emplace_back(import, possible_exports.front());
  }

  return SPV_SUCCESS;
}

spv_result_t CheckImportExportCompatibility(const MessageConsumer& consumer,
                                            const LinkageTable& linkings_to_do,
                                            opt::IRContext* context) {
  spv_position_t position = {};

  // Ensure the import and export types are the same.
  const DecorationManager& decoration_manager = *context->get_decoration_mgr();
  const TypeManager& type_manager = *context->get_type_mgr();
  for (const auto& linking_entry : linkings_to_do) {
    Type* imported_symbol_type =
        type_manager.GetType(linking_entry.imported_symbol.type_id);
    Type* exported_symbol_type =
        type_manager.GetType(linking_entry.exported_symbol.type_id);
    if (!(*imported_symbol_type == *exported_symbol_type))
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Type mismatch on symbol \""
             << linking_entry.imported_symbol.name
             << "\" between imported variable/function %"
             << linking_entry.imported_symbol.id
             << " and exported variable/function %"
             << linking_entry.exported_symbol.id << ".";
  }

  // Ensure the import and export decorations are similar
  for (const auto& linking_entry : linkings_to_do) {
    if (!decoration_manager.HaveTheSameDecorations(
            linking_entry.imported_symbol.id, linking_entry.exported_symbol.id))
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Decorations mismatch on symbol \""
             << linking_entry.imported_symbol.name
             << "\" between imported variable/function %"
             << linking_entry.imported_symbol.id
             << " and exported variable/function %"
             << linking_entry.exported_symbol.id << ".";
    // TODO(pierremoreau): Decorations on function parameters should probably
    //                     match, except for FuncParamAttr if I understand the
    //                     spec correctly.
    // TODO(pierremoreau): Decorations on the function return type should
    //                     match, except for FuncParamAttr.
  }

  return SPV_SUCCESS;
}

spv_result_t RemoveLinkageSpecificInstructions(
    const MessageConsumer& consumer, const LinkerOptions& options,
    const LinkageTable& linkings_to_do, DecorationManager* decoration_manager,
    opt::IRContext* linked_context) {
  spv_position_t position = {};

  if (decoration_manager == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|decoration_manager| of RemoveLinkageSpecificInstructions "
              "should not be empty.";
  if (linked_context == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|linked_module| of RemoveLinkageSpecificInstructions should not "
              "be empty.";

  // TODO(pierremoreau): Remove FuncParamAttr decorations of imported
  // functions' return type.

  // Remove prototypes of imported functions
  for (const auto& linking_entry : linkings_to_do) {
    for (auto func_iter = linked_context->module()->begin();
         func_iter != linked_context->module()->end();) {
      if (func_iter->result_id() == linking_entry.imported_symbol.id)
        func_iter = func_iter.Erase();
      else
        ++func_iter;
    }
  }

  // Remove declarations of imported variables
  for (const auto& linking_entry : linkings_to_do) {
    auto next = linked_context->types_values_begin();
    for (auto inst = next; inst != linked_context->types_values_end();
         inst = next) {
      ++next;
      if (inst->result_id() == linking_entry.imported_symbol.id) {
        linked_context->KillInst(&*inst);
      }
    }
  }

  // If partial linkage is allowed, we need an efficient way to check whether
  // an imported ID had a corresponding export symbol. As uses of the imported
  // symbol have already been replaced by the exported symbol, use the exported
  // symbol ID.
  // TODO(pierremoreau): This will not work if the decoration is applied
  //                     through a group, but the linker does not support that
  //                     either.
  std::unordered_set<SpvId> imports;
  if (options.GetAllowPartialLinkage()) {
    imports.reserve(linkings_to_do.size());
    for (const auto& linking_entry : linkings_to_do)
      imports.emplace(linking_entry.exported_symbol.id);
  }

  // Remove import linkage attributes
  auto next = linked_context->annotation_begin();
  for (auto inst = next; inst != linked_context->annotation_end();
       inst = next) {
    ++next;
    // If this is an import annotation:
    // * if we do not allow partial linkage, remove all import annotations;
    // * otherwise, remove the annotation only if there was a corresponding
    //   export.
    if (inst->opcode() == SpvOpDecorate &&
        inst->GetSingleWordOperand(1u) == SpvDecorationLinkageAttributes &&
        inst->GetSingleWordOperand(3u) == SpvLinkageTypeImport &&
        (!options.GetAllowPartialLinkage() ||
         imports.find(inst->GetSingleWordOperand(0u)) != imports.end())) {
      linked_context->KillInst(&*inst);
    }
  }

  // Remove export linkage attributes if making an executable
  if (!options.GetCreateLibrary()) {
    next = linked_context->annotation_begin();
    for (auto inst = next; inst != linked_context->annotation_end();
         inst = next) {
      ++next;
      if (inst->opcode() == SpvOpDecorate &&
          inst->GetSingleWordOperand(1u) == SpvDecorationLinkageAttributes &&
          inst->GetSingleWordOperand(3u) == SpvLinkageTypeExport) {
        linked_context->KillInst(&*inst);
      }
    }
  }

  // Remove Linkage capability if making an executable and partial linkage is
  // not allowed
  if (!options.GetCreateLibrary() && !options.GetAllowPartialLinkage()) {
    for (auto& inst : linked_context->capabilities())
      if (inst.GetSingleWordInOperand(0u) == SpvCapabilityLinkage) {
        linked_context->KillInst(&inst);
        // The RemoveDuplicatesPass did remove duplicated capabilities, so we
        // now there aren’t more SpvCapabilityLinkage further down.
        break;
      }
  }

  return SPV_SUCCESS;
}

spv_result_t VerifyIds(const MessageConsumer& consumer,
                       opt::IRContext* linked_context) {
  std::unordered_set<uint32_t> ids;
  bool ok = true;
  linked_context->module()->ForEachInst(
      [&ids, &ok](const opt::Instruction* inst) {
        ok &= ids.insert(inst->unique_id()).second;
      });

  if (!ok) {
    consumer(SPV_MSG_INTERNAL_ERROR, "", {}, "Non-unique id in merged module");
    return SPV_ERROR_INVALID_ID;
  }

  return SPV_SUCCESS;
}

}  // namespace

spv_result_t Link(const Context& context,
                  const std::vector<std::vector<uint32_t>>& binaries,
                  std::vector<uint32_t>* linked_binary,
                  const LinkerOptions& options) {
  std::vector<const uint32_t*> binary_ptrs;
  binary_ptrs.reserve(binaries.size());
  std::vector<size_t> binary_sizes;
  binary_sizes.reserve(binaries.size());

  for (const auto& binary : binaries) {
    binary_ptrs.push_back(binary.data());
    binary_sizes.push_back(binary.size());
  }

  return Link(context, binary_ptrs.data(), binary_sizes.data(), binaries.size(),
              linked_binary, options);
}

spv_result_t Link(const Context& context, const uint32_t* const* binaries,
                  const size_t* binary_sizes, size_t num_binaries,
                  std::vector<uint32_t>* linked_binary,
                  const LinkerOptions& options) {
  spv_position_t position = {};
  const spv_context& c_context = context.CContext();
  const MessageConsumer& consumer = c_context->consumer;

  linked_binary->clear();
  if (num_binaries == 0u)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
           << "No modules were given.";

  std::vector<std::unique_ptr<IRContext>> ir_contexts;
  std::vector<Module*> modules;
  modules.reserve(num_binaries);
  for (size_t i = 0u; i < num_binaries; ++i) {
    const uint32_t schema = binaries[i][4u];
    if (schema != 0u) {
      position.index = 4u;
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Schema is non-zero for module " << i + 1 << ".";
    }

    std::unique_ptr<IRContext> ir_context = BuildModule(
        c_context->target_env, consumer, binaries[i], binary_sizes[i]);
    if (ir_context == nullptr)
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Failed to build module " << i + 1 << " out of " << num_binaries
             << ".";
    modules.push_back(ir_context->module());
    ir_contexts.push_back(std::move(ir_context));
  }

  // Phase 1: Shift the IDs used in each binary so that they occupy a disjoint
  //          range from the other binaries, and compute the new ID bound.
  uint32_t max_id_bound = 0u;
  spv_result_t res = ShiftIdsInModules(consumer, &modules, &max_id_bound);
  if (res != SPV_SUCCESS) return res;

  // Phase 2: Remove declarations of constant WorkgroupSize built-ins and
  //          instead define the execution mode for the corresponding kernel.
  for (uint32_t i = 0u; i < num_binaries; ++i) {
    res =
        ConsolidateWorkgroupSize(consumer, i, modules[i], ir_contexts[i].get());
    if (res != SPV_SUCCESS && res != SPV_WARNING) return res;
  }

  // Phase 3: Generate the header
  opt::ModuleHeader header;
  res = GenerateHeader(consumer, modules, max_id_bound, &header);
  if (res != SPV_SUCCESS) return res;
  IRContext linked_context(c_context->target_env, consumer);
  linked_context.module()->SetHeader(header);

  // Phase 4: Merge all the binaries into a single one.
  AssemblyGrammar grammar(c_context);
  res = MergeModules(consumer, modules, grammar, &linked_context);
  if (res != SPV_SUCCESS) return res;

  if (options.GetVerifyIds()) {
    res = VerifyIds(consumer, &linked_context);
    if (res != SPV_SUCCESS) return res;
  }

  // Phase 5: Find the import/export pairs
  LinkageTable linkings_to_do;
  res = GetImportExportPairs(consumer, linked_context,
                             *linked_context.get_def_use_mgr(),
                             *linked_context.get_decoration_mgr(),
                             options.GetAllowPartialLinkage(), &linkings_to_do);
  if (res != SPV_SUCCESS) return res;

  // Phase 6: Ensure the import and export have the same types and decorations.
  res =
      CheckImportExportCompatibility(consumer, linkings_to_do, &linked_context);
  if (res != SPV_SUCCESS) return res;

  // Phase 7: Remove duplicates
  PassManager manager;
  manager.SetMessageConsumer(consumer);
  manager.AddPass<RemoveDuplicatesPass>();
  opt::Pass::Status pass_res = manager.Run(&linked_context);
  if (pass_res == opt::Pass::Status::Failure) return SPV_ERROR_INVALID_DATA;

  // Phase 8: Remove all names and decorations of import variables/functions
  for (const auto& linking_entry : linkings_to_do) {
    linked_context.KillNamesAndDecorates(linking_entry.imported_symbol.id);
    for (const auto parameter_id :
         linking_entry.imported_symbol.parameter_ids) {
      linked_context.KillNamesAndDecorates(parameter_id);
    }
  }

  // Phase 9: Rematch import variables/functions to export variables/functions
  for (const auto& linking_entry : linkings_to_do) {
    linked_context.ReplaceAllUsesWith(linking_entry.imported_symbol.id,
                                      linking_entry.exported_symbol.id);
  }

  // Phase 10: Remove linkage specific instructions, such as import/export
  //           attributes, linkage capability, etc. if applicable
  res = RemoveLinkageSpecificInstructions(consumer, options, linkings_to_do,
                                          linked_context.get_decoration_mgr(),
                                          &linked_context);
  if (res != SPV_SUCCESS) return res;

  // Phase 11: Compact the IDs used in the module
  manager.AddPass<opt::CompactIdsPass>();
  pass_res = manager.Run(&linked_context);
  if (pass_res == opt::Pass::Status::Failure) return SPV_ERROR_INVALID_DATA;

  // Phase 12: Output the module
  linked_context.module()->ToBinary(linked_binary, true);

  return SPV_SUCCESS;
}

}  // namespace spvtools
