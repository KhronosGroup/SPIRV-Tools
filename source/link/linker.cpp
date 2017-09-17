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

#include <cstdio>
#include <cstring>

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "assembly_grammar.h"
#include "diagnostic.h"
#include "opt/build_module.h"
#include "opt/compact_ids_pass.h"
#include "opt/decoration_manager.h"
#include "opt/ir_loader.h"
#include "opt/make_unique.h"
#include "opt/pass_manager.h"
#include "opt/remove_duplicates_pass.h"
#include "spirv-tools/libspirv.hpp"
#include "spirv_target_env.h"

namespace spvtools {

using ir::Instruction;
using ir::Module;
using ir::Operand;
using opt::PassManager;
using opt::RemoveDuplicatesPass;
using opt::analysis::DecorationManager;
using opt::analysis::DefUseManager;

// Stores various information about an imported or exported symbol.
struct LinkageSymbolInfo {
  SpvId id;          // ID of the symbol
  SpvId typeId;      // ID of the type of the symbol
  std::string name;  // unique name defining the symbol and used for matching
                     // imports and exports together
  std::vector<SpvId> parametersIds;  // ID of the parameters of the symbol, if
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
// not be empty either.
static spv_result_t ShiftIdsInModules(
    const MessageConsumer& consumer,
    std::vector<std::unique_ptr<ir::Module>>* modules, uint32_t* max_id_bound);

// Generates the header for the linked module and returns it in |header|.
//
// |header| should not be null, |modules| should not be empty and
// |max_id_bound| should be strictly greater than 0.
//
// TODO(pierremoreau): What to do when binaries use different versions of
//                     SPIR-V? For now, use the max of all versions found in
//                     the input modules.
static spv_result_t GenerateHeader(
    const MessageConsumer& consumer,
    const std::vector<std::unique_ptr<ir::Module>>& modules,
    uint32_t max_id_bound, ir::ModuleHeader* header);

// Merge all the modules from |inModules| into |linked_module|.
//
// |linked_module| should not be null.
static spv_result_t MergeModules(
    const MessageConsumer& consumer,
    const std::vector<std::unique_ptr<Module>>& inModules,
    const libspirv::AssemblyGrammar& grammar, Module* linked_module);

// Compute all pairs of import and export and return it in |linkingsToDo|.
//
// |linkingsToDo should not be null.
//
// TODO(pierremoreau): Linkage attributes applied by a group decoration are
//                     currently not handled. (You could have a group being
//                     applied to a single ID.)
static spv_result_t GetImportExportPairs(const MessageConsumer& consumer,
                                         const Module& linked_module,
                                         const DefUseManager& defUseManager,
                                         LinkageTable* linkingsToDo);

// Checks that for each pair of import and export, the import and export have
// the same type as well as the same decorations.
//
// TODO(pierremoreau): Decorations on functions parameters are currently not
// checked.
static spv_result_t CheckImportExportCompatibility(
    const MessageConsumer& consumer, const LinkageTable& linkingsToDo,
    const DefUseManager& defUseManager,
    const DecorationManager& decorationManager);

// Remove linkage specific instructions, such as prototypes of imported
// functions, declarations of imported variables, import (and export if
// necessary) linkage attribtes.
//
// |linked_module| and |decorationManager| should not be null, and the
// 'RemoveDuplicatePass' should be run first.
//
// TODO(pierremoreau): Linkage attributes applied by a group decoration are
//                     currently not handled. (You could have a group being
//                     applied to a single ID.)
// TODO(pierremoreau): Run a pass for removing dead instructions, for example
//                     OpName for prototypes of imported funcions.
static spv_result_t RemoveLinkageSpecificInstructions(
    const MessageConsumer& consumer, bool create_executable,
    const LinkageTable& linkingsToDo, DecorationManager* decorationManager,
    Module* linked_module);

// Structs for holding the data members for SpvLinker.
struct Linker::Impl {
  explicit Impl(spv_target_env env) : context(spvContextCreate(env)) {
    // The default consumer in spv_context_t is a null consumer, which provides
    // equivalent functionality (from the user's perspective) as a real consumer
    // does nothing.
  }
  ~Impl() { spvContextDestroy(context); }

  spv_context context;  // C interface context object.
};

Linker::Linker(spv_target_env env) : impl_(new Impl(env)) {}

Linker::~Linker() {}

void Linker::SetMessageConsumer(MessageConsumer consumer) {
  SetContextMessageConsumer(impl_->context, std::move(consumer));
}

spv_result_t Linker::Link(const std::vector<std::vector<uint32_t>>& binaries,
                          std::vector<uint32_t>& linked_binary,
                          const LinkerOptions& options) const {
  spv_position_t position = {};
  const MessageConsumer& consumer = impl_->context->consumer;

  linked_binary.clear();
  if (binaries.empty())
    return libspirv::DiagnosticStream(position, consumer,
                                      SPV_ERROR_INVALID_BINARY)
           << "No modules were given.";

  std::vector<std::unique_ptr<Module>> modules;
  modules.reserve(binaries.size());
  for (const auto& mod : binaries) {
    std::unique_ptr<Module> module = BuildModule(
        impl_->context->target_env, consumer, mod.data(), mod.size());
    if (module == nullptr)
      return libspirv::DiagnosticStream(position, consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "Failed to build a module out of " << modules.size() << ".";
    modules.push_back(std::move(module));
  }

  // Phase 1: Shift the IDs used in each binary so that they occupy a disjoint
  //          range from the other binaries, and compute the new ID bound.
  uint32_t max_id_bound;
  spv_result_t res = ShiftIdsInModules(consumer, &modules, &max_id_bound);
  if (res != SPV_SUCCESS) return res;

  // Phase 2: Generate the header
  ir::ModuleHeader header;
  res = GenerateHeader(consumer, modules, max_id_bound, &header);
  if (res != SPV_SUCCESS) return res;
  auto linked_module = MakeUnique<Module>();
  linked_module->SetHeader(header);

  // Phase 3: Merge all the binaries into a single one.
  libspirv::AssemblyGrammar grammar(impl_->context);
  res = MergeModules(consumer, modules, grammar, linked_module.get());
  if (res != SPV_SUCCESS) return res;

  DefUseManager defUseManager(consumer, linked_module.get());

  // Phase 4: Find the import/export pairs
  LinkageTable linkingsToDo;
  res = GetImportExportPairs(consumer, *linked_module, defUseManager,
                             &linkingsToDo);
  if (res != SPV_SUCCESS) return res;

  // Phase 5: Ensure the import and export have the same types and decorations.
  DecorationManager decorationManager(linked_module.get());
  res = CheckImportExportCompatibility(consumer, linkingsToDo, defUseManager,
                                       decorationManager);
  if (res != SPV_SUCCESS) return res;

  // Phase 6: Remove duplicates
  PassManager manager;
  manager.SetMessageConsumer(consumer);
  manager.AddPass<RemoveDuplicatesPass>();
  opt::Pass::Status pass_res = manager.Run(linked_module.get());
  if (pass_res == opt::Pass::Status::Failure) return SPV_ERROR_INVALID_DATA;

  // Phase 7: Remove linkage specific instructions, such as import/export
  // attributes, linkage capability, etc. if applicable
  res = RemoveLinkageSpecificInstructions(consumer, !options.GetCreateLibrary(),
                                          linkingsToDo, &decorationManager,
                                          linked_module.get());
  if (res != SPV_SUCCESS) return res;

  // Phase 8: Rematch import variables/functions to export variables/functions
  // TODO(pierremoreau): Keep the previous DefUseManager up-to-date
  DefUseManager def_use_manager2(consumer, linked_module.get());
  for (const auto& linking_entry : linkingsToDo)
    def_use_manager2.ReplaceAllUsesWith(linking_entry.imported_symbol.id,
                                        linking_entry.exported_symbol.id);

  // Phase 9: Compact the IDs used in the module
  manager.AddPass<opt::CompactIdsPass>();
  pass_res = manager.Run(linked_module.get());
  if (pass_res == opt::Pass::Status::Failure) return SPV_ERROR_INVALID_DATA;

  // Phase 10: Output the module
  linked_module->ToBinary(&linked_binary, true);

  return SPV_SUCCESS;
}

spv_result_t Linker::Link(const uint32_t* const* binaries,
                          const size_t* binary_sizes, size_t num_binaries,
                          std::vector<uint32_t>& linked_binary,
                          const LinkerOptions& options) const {
  std::vector<std::vector<uint32_t>> binaries_array;
  binaries_array.reserve(num_binaries);
  for (size_t i = 0u; i < num_binaries; ++i)
    binaries_array.push_back({binaries[i], binaries[i] + binary_sizes[i]});

  return Link(binaries_array, linked_binary, options);
}

static spv_result_t ShiftIdsInModules(
    const MessageConsumer& consumer,
    std::vector<std::unique_ptr<ir::Module>>* modules, uint32_t* max_id_bound) {
  spv_position_t position = {};
  if (modules == nullptr)
    return libspirv::DiagnosticStream(position, consumer,
                                      SPV_ERROR_INVALID_DATA)
           << "|modules| of ShiftIdsInModules should not be null.";
  if (modules->empty())
    return libspirv::DiagnosticStream(position, consumer,
                                      SPV_ERROR_INVALID_DATA)
           << "|modules| of ShiftIdsInModules should not be empty.";
  if (max_id_bound == nullptr)
    return libspirv::DiagnosticStream(position, consumer,
                                      SPV_ERROR_INVALID_DATA)
           << "|max_id_bound| of ShiftIdsInModules should not be null.";

  uint32_t id_bound = modules->front()->IdBound() - 1u;
  for (auto module_iter = modules->begin() + 1; module_iter != modules->end();
       ++module_iter) {
    Module* module = module_iter->get();
    module->ForEachInst([&id_bound](Instruction* insn) {
      insn->ForEachId([&id_bound](uint32_t* id) { *id += id_bound; });
    });
    id_bound += module->IdBound() - 1u;
    if (id_bound > 0x3FFFFF)
      return libspirv::DiagnosticStream(position, consumer,
                                        SPV_ERROR_INVALID_ID)
             << "The limit of IDs, 4194303, was exceeded:"
             << " " << id_bound << " is the current ID bound.";
  }
  ++id_bound;
  if (id_bound > 0x3FFFFF)
    return libspirv::DiagnosticStream(position, consumer, SPV_ERROR_INVALID_ID)
           << "The limit of IDs, 4194303, was exceeded:"
           << " " << id_bound << " is the current ID bound.";

  *max_id_bound = id_bound;

  return SPV_SUCCESS;
}

static spv_result_t GenerateHeader(
    const MessageConsumer& consumer,
    const std::vector<std::unique_ptr<ir::Module>>& modules,
    uint32_t max_id_bound, ir::ModuleHeader* header) {
  spv_position_t position = {};

  if (modules.empty())
    return libspirv::DiagnosticStream(position, consumer,
                                      SPV_ERROR_INVALID_DATA)
           << "|modules| of GenerateHeader should not be empty.";
  if (max_id_bound == 0u)
    return libspirv::DiagnosticStream(position, consumer,
                                      SPV_ERROR_INVALID_DATA)
           << "|max_id_bound| of GenerateHeader should not be null.";

  uint32_t version = 0u;
  for (const auto& module : modules)
    version = std::max(version, module->version());

  header->magic_number = SpvMagicNumber;
  header->version = version;
  header->generator = 17u;
  header->bound = max_id_bound;
  header->reserved = 0u;

  return SPV_SUCCESS;
}

static spv_result_t MergeModules(
    const MessageConsumer& consumer,
    const std::vector<std::unique_ptr<Module>>& inModules,
    const libspirv::AssemblyGrammar& grammar, Module* linked_module) {
  spv_position_t position = {};

  if (linked_module == nullptr)
    return libspirv::DiagnosticStream(position, consumer,
                                      SPV_ERROR_INVALID_DATA)
           << "|linked_module| of MergeModules should not be null.";

  if (inModules.empty()) return SPV_SUCCESS;

  for (const auto& module : inModules)
    for (const auto& insn : module->capabilities())
      linked_module->AddCapability(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->extensions())
      linked_module->AddExtension(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->ext_inst_imports())
      linked_module->AddExtInstImport(MakeUnique<Instruction>(insn));

  do {
    const Instruction* memoryModelInsn = inModules[0]->GetMemoryModel();
    if (memoryModelInsn == nullptr) break;

    uint32_t addressingModel = memoryModelInsn->GetSingleWordOperand(0u);
    uint32_t memoryModel = memoryModelInsn->GetSingleWordOperand(1u);
    for (const auto& module : inModules) {
      memoryModelInsn = module->GetMemoryModel();
      if (memoryModelInsn == nullptr) continue;

      if (addressingModel != memoryModelInsn->GetSingleWordOperand(0u)) {
        spv_operand_desc initialDesc = nullptr, currentDesc = nullptr;
        grammar.lookupOperand(SPV_OPERAND_TYPE_ADDRESSING_MODEL,
                              addressingModel, &initialDesc);
        grammar.lookupOperand(SPV_OPERAND_TYPE_ADDRESSING_MODEL,
                              memoryModelInsn->GetSingleWordOperand(0u),
                              &currentDesc);
        return libspirv::DiagnosticStream(position, consumer,
                                          SPV_ERROR_INTERNAL)
               << "Conflicting addressing models: " << initialDesc->name
               << " vs " << currentDesc->name << ".";
      }
      if (memoryModel != memoryModelInsn->GetSingleWordOperand(1u)) {
        spv_operand_desc initialDesc = nullptr, currentDesc = nullptr;
        grammar.lookupOperand(SPV_OPERAND_TYPE_MEMORY_MODEL, memoryModel,
                              &initialDesc);
        grammar.lookupOperand(SPV_OPERAND_TYPE_MEMORY_MODEL,
                              memoryModelInsn->GetSingleWordOperand(1u),
                              &currentDesc);
        return libspirv::DiagnosticStream(position, consumer,
                                          SPV_ERROR_INTERNAL)
               << "Conflicting memory models: " << initialDesc->name << " vs "
               << currentDesc->name << ".";
      }
    }

    if (memoryModelInsn != nullptr)
      linked_module->SetMemoryModel(MakeUnique<Instruction>(*memoryModelInsn));
  } while (false);

  std::vector<std::pair<uint32_t, const char*>> entryPoints;
  for (const auto& module : inModules)
    for (const auto& insn : module->entry_points()) {
      const uint32_t model = insn.GetSingleWordInOperand(0);
      const char* const name =
          reinterpret_cast<const char*>(insn.GetInOperand(2).words.data());
      const auto i = std::find_if(
          entryPoints.begin(), entryPoints.end(),
          [model, name](const std::pair<uint32_t, const char*>& v) {
            return v.first == model && strcmp(name, v.second) == 0;
          });
      if (i != entryPoints.end()) {
        spv_operand_desc desc = nullptr;
        grammar.lookupOperand(SPV_OPERAND_TYPE_EXECUTION_MODEL, model, &desc);
        return libspirv::DiagnosticStream(position, consumer,
                                          SPV_ERROR_INTERNAL)
               << "The entry point \"" << name << "\", with execution model "
               << desc->name << ", was already defined.";
      }
      linked_module->AddEntryPoint(MakeUnique<Instruction>(insn));
      entryPoints.emplace_back(model, name);
    }

  for (const auto& module : inModules)
    for (const auto& insn : module->execution_modes())
      linked_module->AddExecutionMode(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->debugs1())
      linked_module->AddDebug1Inst(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->debugs2())
      linked_module->AddDebug2Inst(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->annotations())
      linked_module->AddAnnotationInst(MakeUnique<Instruction>(insn));

  // TODO(pierremoreau): Since the modules have not been validate, should we
  //                     expect SpvStorageClassFunction variables outside
  //                     functions?
  uint32_t num_global_values = 0u;
  for (const auto& module : inModules) {
    for (const auto& insn : module->types_values()) {
      linked_module->AddType(MakeUnique<Instruction>(insn));
      num_global_values += insn.opcode() == SpvOpVariable;
    }
  }
  if (num_global_values > 0xFFFF)
    return libspirv::DiagnosticStream(position, consumer, SPV_ERROR_INTERNAL)
           << "The limit of global values, 65535, was exceeded;"
           << " " << num_global_values << " global values were found.";

  // Process functions and their basic blocks
  for (const auto& module : inModules) {
    for (const auto& func : *module) {
      std::unique_ptr<ir::Function> cloned_func =
          MakeUnique<ir::Function>(func);
      cloned_func->SetParent(linked_module);
      linked_module->AddFunction(std::move(cloned_func));
    }
  }

  return SPV_SUCCESS;
}

static spv_result_t GetImportExportPairs(const MessageConsumer& consumer,
                                         const Module& linked_module,
                                         const DefUseManager& defUseManager,
                                         LinkageTable* linkingsToDo) {
  spv_position_t position = {};

  if (linkingsToDo == nullptr)
    return libspirv::DiagnosticStream(position, consumer,
                                      SPV_ERROR_INVALID_DATA)
           << "|linkingsToDo| of GetImportExportPairs should not be empty.";

  std::vector<LinkageSymbolInfo> imports;
  std::unordered_map<std::string, std::vector<LinkageSymbolInfo>> exports;

  // Figure out the imports and exports
  for (const auto& decoration : linked_module.annotations()) {
    if (decoration.opcode() != SpvOpDecorate ||
        decoration.GetSingleWordInOperand(1u) != SpvDecorationLinkageAttributes)
      continue;

    const uint32_t type = decoration.GetSingleWordInOperand(3u);
    const SpvId id = decoration.GetSingleWordInOperand(0u);

    LinkageSymbolInfo data;
    data.name =
        reinterpret_cast<const char*>(decoration.GetInOperand(2u).words.data());
    data.id = id;
    data.typeId = 0u;

    // Retrieve the type of the current symbol. This information will be used
    // when checking that the imported and exported symbols have the same
    // types.
    const Instruction* def_inst = defUseManager.GetDef(id);
    if (def_inst == nullptr)
      return libspirv::DiagnosticStream(position, consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "ID " << id << " is never defined:\n";

    if (def_inst->opcode() == SpvOpVariable) {
      data.typeId = def_inst->type_id();
    } else if (def_inst->opcode() == SpvOpFunction) {
      data.typeId = def_inst->GetSingleWordInOperand(1u);

      // range-based for loop calls begin()/end(), but never cbegin()/cend(),
      // which will not work here.
      for (auto func_iter = linked_module.cbegin();
           func_iter != linked_module.cend(); ++func_iter) {
        if (func_iter->result_id() != id) continue;
        func_iter->ForEachParam([&data](const Instruction* inst) {
          data.parametersIds.push_back(inst->result_id());
        });
      }
    } else {
      return libspirv::DiagnosticStream(position, consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "Only global variables and functions can be decorated using"
             << " LinkageAttributes; " << id << " is neither of them.\n";
    }

    if (type == SpvLinkageTypeImport)
      imports.push_back(data);
    else if (type == SpvLinkageTypeExport)
      exports[data.name].push_back(data);
  }

  // Find the import/export pairs
  for (const auto& import : imports) {
    std::vector<LinkageSymbolInfo> possibleExports;
    const auto& exp = exports.find(import.name);
    if (exp != exports.end()) possibleExports = exp->second;
    if (possibleExports.empty())
      return libspirv::DiagnosticStream(position, consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "No export linkage was found for \"" << import.name << "\".";
    else if (possibleExports.size() > 1u)
      return libspirv::DiagnosticStream(position, consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "Too many export linkages, " << possibleExports.size()
             << ", were found for \"" << import.name << "\".";

    linkingsToDo->emplace_back(import, possibleExports.front());
  }

  return SPV_SUCCESS;
}

static spv_result_t CheckImportExportCompatibility(
    const MessageConsumer& consumer, const LinkageTable& linkingsToDo,
    const DefUseManager& defUseManager,
    const DecorationManager& decorationManager) {
  spv_position_t position = {};

  // Ensure th import and export types are the same.
  for (const auto& linking_entry : linkingsToDo) {
    if (!RemoveDuplicatesPass::AreTypesEqual(
            *defUseManager.GetDef(linking_entry.imported_symbol.typeId),
            *defUseManager.GetDef(linking_entry.exported_symbol.typeId),
            defUseManager, decorationManager))
      return libspirv::DiagnosticStream(position, consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "Type mismatch between imported variable/function %"
             << linking_entry.imported_symbol.id
             << " and exported variable/function %"
             << linking_entry.exported_symbol.id << ".";
  }

  // Ensure the import and export decorations are similar
  for (const auto& linking_entry : linkingsToDo) {
    if (!decorationManager.HaveTheSameDecorations(
            linking_entry.imported_symbol.id, linking_entry.exported_symbol.id))
      return libspirv::DiagnosticStream(position, consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "Decorations mismatch between imported variable/function %"
             << linking_entry.imported_symbol.id
             << " and exported variable/function %"
             << linking_entry.exported_symbol.id << ".";
    // TODO(pierremoreau): Decorations on function parameters should probably
    //                     match, except for FuncParamAttr if I understand the
    //                     spec correctly, which makes the code more
    //                     complicated.
    //    for (uint32_t i = 0u; i <
    //    linking_entry.imported_symbol.parametersIds.size(); ++i)
    //      if
    //      (!decorationManager.HaveTheSameDecorations(linking_entry.imported_symbol.parametersIds[i],
    //      linking_entry.exported_symbol.parametersIds[i]))
    //          return libspirv::DiagnosticStream(position,
    //          impl_->context->consumer,
    //                                            SPV_ERROR_INVALID_BINARY)
    //                 << "Decorations mismatch between imported function %" <<
    //                 linking_entry.imported_symbol.id << "'s"
    //                 << " and exported function %" <<
    //                 linking_entry.exported_symbol.id << "'s " << (i + 1u) <<
    //                 "th parameter.";
  }

  return SPV_SUCCESS;
}

static spv_result_t RemoveLinkageSpecificInstructions(
    const MessageConsumer& consumer, bool create_executable,
    const LinkageTable& linkingsToDo, DecorationManager* decorationManager,
    Module* linked_module) {
  spv_position_t position = {};

  if (decorationManager == nullptr)
    return libspirv::DiagnosticStream(position, consumer,
                                      SPV_ERROR_INVALID_DATA)
           << "|decorationManager| of RemoveLinkageSpecificInstructions should "
              "not "
              "be empty.";
  if (linked_module == nullptr)
    return libspirv::DiagnosticStream(position, consumer,
                                      SPV_ERROR_INVALID_DATA)
           << "|linked_module| of RemoveLinkageSpecificInstructions should not "
              "be empty.";

  // Remove FuncParamAttr decorations of imported functions' parameters.
  // From the SPIR-V specification, Sec. 2.13:
  //   When resolving imported functions, the Function Control and all Function
  //   Parameter Attributes are taken from the function definition, and not
  //   from the function declaration.
  for (const auto& linking_entry : linkingsToDo) {
    for (const auto parameter_id :
         linking_entry.imported_symbol.parametersIds) {
      for (ir::Instruction* decoration :
           decorationManager->GetDecorationsFor(parameter_id, false)) {
        switch (decoration->opcode()) {
          case SpvOpDecorate:
          case SpvOpMemberDecorate:
            if (decoration->GetSingleWordInOperand(1u) ==
                SpvDecorationFuncParamAttr)
              decoration->ToNop();
            break;
          default:
            break;
        }
      }
    }
  }

  // Remove prototypes of imported functions
  for (const auto& linking_entry : linkingsToDo) {
    for (auto func_iter = linked_module->begin();
         func_iter != linked_module->end();) {
      if (func_iter->result_id() == linking_entry.imported_symbol.id)
        func_iter = func_iter.Erase();
      else
        ++func_iter;
    }
  }

  // Remove declarations of imported variables
  for (const auto& linking_entry : linkingsToDo) {
    for (auto& inst : linked_module->types_values())
      if (inst.result_id() == linking_entry.imported_symbol.id) inst.ToNop();
  }

  // Remove import linkage attributes
  for (auto& inst : linked_module->annotations())
    if (inst.opcode() == SpvOpDecorate &&
        inst.GetSingleWordOperand(1u) == SpvDecorationLinkageAttributes &&
        inst.GetSingleWordOperand(3u) == SpvLinkageTypeImport)
      inst.ToNop();

  // Remove export linkage attributes and Linkage capability if making an
  // executable
  if (create_executable) {
    for (auto& inst : linked_module->annotations())
      if (inst.opcode() == SpvOpDecorate &&
          inst.GetSingleWordOperand(1u) == SpvDecorationLinkageAttributes &&
          inst.GetSingleWordOperand(3u) == SpvLinkageTypeExport)
        inst.ToNop();

    for (auto& inst : linked_module->capabilities())
      if (inst.GetSingleWordInOperand(0u) == SpvCapabilityLinkage) {
        inst.ToNop();
        // The RemoveDuplicatesPass did remove duplicated capabilities, so we
        // now there aren’t more SpvCapabilityLinkage further down.
        break;
      }
  }

  return SPV_SUCCESS;
}

}  // namespace spvtools
