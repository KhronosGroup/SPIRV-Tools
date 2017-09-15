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
using opt::analysis::DefUseManager;

struct LinkingInfo {
  SpvId id;
  SpvId typeId;
  std::string name;
  std::vector<SpvId> parametersIds;
};

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

// Merge all the modules from |inModules| into |linkedModule|.
//
// |linkedModule| should not be null.
static spv_result_t MergeModules(
    const MessageConsumer& consumer,
    const std::vector<std::unique_ptr<Module>>& inModules,
    const libspirv::AssemblyGrammar& grammar, Module* linkedModule);

// Compute all pairs of import and export and return it in |linkingsToDo|.
//
// |linkingsToDo should not be null.
static spv_result_t GetImportExportPairs(
    const MessageConsumer& consumer, const Module& linkedModule,
    const DefUseManager& defUseManager,
    std::vector<std::pair<LinkingInfo, LinkingInfo>>* linkingsToDo);

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

  PassManager manager;
  manager.SetMessageConsumer(consumer);

  // Phase 1: Shift the IDs used in each binary so that they occupy a disjoint
  //          range from the other binaries, and compute the new ID bound.
  uint32_t max_id_bound;
  spv_result_t res = ShiftIdsInModules(consumer, &modules, &max_id_bound);
  if (res != SPV_SUCCESS) return res;

  // Phase 2: Generate the header
  ir::ModuleHeader header;
  res = GenerateHeader(consumer, modules, max_id_bound, &header);
  if (res != SPV_SUCCESS) return res;
  auto linkedModule = MakeUnique<Module>();
  linkedModule->SetHeader(header);

  // Phase 3: Merge all the binaries into a single one.
  libspirv::AssemblyGrammar grammar(impl_->context);
  res = MergeModules(consumer, modules, grammar, linkedModule.get());
  if (res != SPV_SUCCESS) return res;

  DefUseManager defUseManager(consumer, linkedModule.get());

  // Find the import/export pairs
  std::vector<std::pair<LinkingInfo, LinkingInfo>> linkingsToDo;
  res = GetImportExportPairs(consumer, *linkedModule, defUseManager,
                             &linkingsToDo);
  if (res != SPV_SUCCESS) return res;

  // Ensure the import and export types are similar
  opt::analysis::DecorationManager decorationManager(linkedModule.get());
  for (const auto& i : linkingsToDo) {
    if (!RemoveDuplicatesPass::AreTypesEqual(
            *defUseManager.GetDef(i.first.typeId),
            *defUseManager.GetDef(i.second.typeId), defUseManager,
            decorationManager))
      return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "Type mismatch between imported variable/function %"
             << i.first.id << " and exported variable/function %" << i.second.id
             << ".";
  }

  // Ensure the import and export decorations are similar
  for (const auto& i : linkingsToDo) {
    if (!decorationManager.HaveTheSameDecorations(i.first.id, i.second.id))
      return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "Decorations mismatch between imported variable/function %"
             << i.first.id << " and exported variable/function %" << i.second.id
             << ".";
    // TODO(pierremoreau): Decorations on function parameters should probably
    //                     match, except for FuncParamAttr if I understand the
    //                     spec correctly, which makes the code more
    //                     complicated.
    //    for (uint32_t j = 0u; j < i.first.parametersIds.size(); ++j)
    //      if
    //      (!decorationManager.HaveTheSameDecorations(i.first.parametersIds[j],
    //      i.second.parametersIds[j]))
    //          return libspirv::DiagnosticStream(position,
    //          impl_->context->consumer,
    //                                            SPV_ERROR_INVALID_BINARY)
    //                 << "Decorations mismatch between imported function %" <<
    //                 i.first.id << "'s"
    //                 << " and exported function %" << i.second.id << "'s " <<
    //                 (j + 1u) << "th parameter.";
  }

  // Remove FuncParamAttr decorations of imported functions' parameters.
  // From the SPIR-V specification, Sec. 2.13:
  //   When resolving imported functions, the Function Control and all Function
  //   Parameter Attributes are taken from the function definition, and not
  //   from the function declaration.
  for (const auto& i : linkingsToDo) {
    for (const auto j : i.first.parametersIds) {
      for (ir::Instruction* decoration :
           decorationManager.GetDecorationsFor(j, false)) {
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
  for (const auto& i : linkingsToDo) {
    for (auto j = linkedModule->begin(); j != linkedModule->end();)
      j = (j->result_id() == i.first.id) ? j.Erase() : ++j;
  }

  // Remove declarations of imported variables
  for (const auto& i : linkingsToDo) {
    for (auto j = linkedModule->types_values_begin();
         j != linkedModule->types_values_end();)
      j = (j->result_id() == i.first.id) ? j.Erase() : ++j;
  }

  // Rematch import variables/functions to export variables/functions
  linkedModule->ForEachInst([&linkingsToDo](Instruction* insn) {
    insn->ForEachId([&linkingsToDo](uint32_t* id) {
      auto id_iter =
          std::find_if(linkingsToDo.begin(), linkingsToDo.end(),
                       [id](const std::pair<LinkingInfo, LinkingInfo>& pair) {
                         return pair.second.id == *id;
                       });
      if (id_iter != linkingsToDo.end()) *id = id_iter->first.id;
    });
  });

  // Remove import linkage attributes
  for (auto i = linkedModule->annotation_begin();
       i != linkedModule->annotation_end();) {
    if (i->opcode() != SpvOpDecorate ||
        i->GetSingleWordOperand(1u) != SpvDecorationLinkageAttributes ||
        i->GetSingleWordOperand(3u) != SpvLinkageTypeImport)
      ++i;
    else
      i = i.Erase();
  }

  // Remove export linkage attributes and Linkage capability if making an
  // executable
  if (!options.GetCreateLibrary()) {
    for (auto i = linkedModule->annotation_begin();
         i != linkedModule->annotation_end();) {
      if (i->opcode() != SpvOpDecorate ||
          i->GetSingleWordOperand(1u) != SpvDecorationLinkageAttributes ||
          i->GetSingleWordOperand(3u) != SpvLinkageTypeExport)
        ++i;
      else
        i = i.Erase();
    }

    // Remove duplicates
    manager.AddPass<RemoveDuplicatesPass>();

    for (auto i = linkedModule->capability_begin();
         i != linkedModule->capability_end();) {
      if (i->GetSingleWordInOperand(0u) != SpvCapabilityLinkage)
        ++i;
      else
        i = i.Erase();
    }
  }

  // Queue an optimisation pass to compact all IDs.
  manager.AddPass<opt::CompactIdsPass>();

  // Phase 6: Run all accumulated optimisation passes.
  manager.Run(linkedModule.get());

  // Phase 7: Output the module
  linkedModule->ToBinary(&linked_binary, true);

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
  for (auto i = modules->begin() + 1; i != modules->end(); ++i) {
    Module* module = i->get();
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
    const libspirv::AssemblyGrammar& grammar, Module* linkedModule) {
  spv_position_t position = {};

  if (linkedModule == nullptr)
    return libspirv::DiagnosticStream(position, consumer,
                                      SPV_ERROR_INVALID_DATA)
           << "|linkedModule| of MergeModules should not be null.";

  if (inModules.empty()) return SPV_SUCCESS;

  for (const auto& module : inModules)
    for (const auto& insn : module->capabilities())
      linkedModule->AddCapability(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->extensions())
      linkedModule->AddExtension(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->ext_inst_imports())
      linkedModule->AddExtInstImport(MakeUnique<Instruction>(insn));

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
      linkedModule->SetMemoryModel(MakeUnique<Instruction>(*memoryModelInsn));
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
      linkedModule->AddEntryPoint(MakeUnique<Instruction>(insn));
      entryPoints.emplace_back(model, name);
    }

  for (const auto& module : inModules)
    for (const auto& insn : module->execution_modes())
      linkedModule->AddExecutionMode(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->debugs1())
      linkedModule->AddDebug1Inst(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->debugs2())
      linkedModule->AddDebug2Inst(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->annotations())
      linkedModule->AddAnnotationInst(MakeUnique<Instruction>(insn));

  // TODO(pierremoreau): Since the modules have not been validate, should we
  //                     expect SpvStorageClassFunction variables outside
  //                     functions?
  uint32_t num_global_values = 0u;
  for (const auto& module : inModules) {
    for (const auto& insn : module->types_values()) {
      linkedModule->AddType(MakeUnique<Instruction>(insn));
      num_global_values += insn.opcode() == SpvOpVariable;
    }
  }
  if (num_global_values > 0xFFFF)
    return libspirv::DiagnosticStream(position, consumer, SPV_ERROR_INTERNAL)
           << "The limit of global values, 65535, was exceeded;"
           << " " << num_global_values << " global values were found.";

  // Process functions and their basic blocks
  for (const auto& module : inModules) {
    for (const auto& i : *module) {
      std::unique_ptr<ir::Function> func = MakeUnique<ir::Function>(i);
      func->SetParent(linkedModule);
      linkedModule->AddFunction(std::move(func));
    }
  }

  return SPV_SUCCESS;
}

static spv_result_t GetImportExportPairs(
    const MessageConsumer& consumer, const Module& linkedModule,
    const DefUseManager& defUseManager,
    std::vector<std::pair<LinkingInfo, LinkingInfo>>* linkingsToDo) {
  spv_position_t position = {};

  if (linkingsToDo == nullptr)
    return libspirv::DiagnosticStream(position, consumer,
                                      SPV_ERROR_INVALID_DATA)
           << "|linkingsToDo| of GetImportExportPairs should not be empty.";

  std::vector<LinkingInfo> imports;
  std::unordered_map<std::string, std::vector<LinkingInfo>> exports;
  // Figure out the imports and exports
  for (const auto& j : linkedModule.annotations()) {
    if (j.opcode() == SpvOpDecorate &&
        j.GetSingleWordInOperand(1u) == SpvDecorationLinkageAttributes) {
      uint32_t type = j.GetSingleWordInOperand(3u);
      SpvId id = j.GetSingleWordInOperand(0u);
      LinkingInfo data;
      data.name =
          reinterpret_cast<const char*>(j.GetInOperand(2u).words.data());
      data.id = id;
      data.typeId = 0u;
      const Instruction* defInst = defUseManager.GetDef(id);
      if (defInst == nullptr)
        return libspirv::DiagnosticStream(position, consumer,
                                          SPV_ERROR_INVALID_BINARY)
               << "ID " << id << " is never defined:\n";
      if (defInst->opcode() == SpvOpVariable) {
        data.typeId = defInst->type_id();
      } else if (defInst->opcode() == SpvOpFunction) {
        data.typeId = defInst->GetSingleWordInOperand(1u);
        // range-based for loop calls begin()/end(), but never cbegin()/cend(),
        // which will not work here.
        for (auto func_iter = linkedModule.cbegin();
             func_iter != linkedModule.cend(); ++func_iter) {
          if (func_iter->result_id() != id) continue;
          func_iter->ForEachParam([&data](const Instruction* inst) {
            data.parametersIds.push_back(inst->result_id());
          });
        }
      }
      if (type == SpvLinkageTypeImport)
        imports.push_back(data);
      else if (type == SpvLinkageTypeExport)
        exports[data.name].push_back(data);
    }
  }

  // Find the import/export pairs
  for (const auto& import : imports) {
    std::vector<LinkingInfo> possibleExports;
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

}  // namespace spvtools
