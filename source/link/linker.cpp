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

struct TypeData {
  size_t moduleIndex;
  Instruction i;
};

using IdDecorationsList = std::unordered_map<uint32_t, std::vector<Instruction>>;

static IdDecorationsList getDecorationsForId(SpvId id, const std::unique_ptr<Module>& module) {
  IdDecorationsList decorations;
  auto decoIter = module->annotation_end();

  std::vector<std::pair<SpvId, uint32_t>> idsToLookFor;
  idsToLookFor.emplace_back(id, std::numeric_limits<uint32_t>::max());

  // pierremoreau: Assume that OpGroupDecorate can't target an
  //               OpDecorationGroup.
  do {
    --decoIter;
    if (decoIter->opcode() == SpvOpGroupDecorate) {
      for (uint32_t j = 1u; j < decoIter->NumInOperands(); ++j)
        if (decoIter->GetSingleWordInOperand(j) == id) {
          idsToLookFor.emplace_back(decoIter->GetSingleWordInOperand(0u), std::numeric_limits<uint32_t>::max());
          break;
        }
    } else if (decoIter->opcode() == SpvOpGroupMemberDecorate) {
      for (uint32_t j = 1u; j < decoIter->NumInOperands(); j += 2u)
        if (decoIter->GetSingleWordInOperand(j) == id) {
          idsToLookFor.emplace_back(decoIter->GetSingleWordInOperand(0u), decoIter->GetSingleWordInOperand(j + 1u));
          break;
        }
    } else if (decoIter->opcode() == SpvOpMemberDecorate) {
      if (decoIter->GetSingleWordInOperand(0u) == id)
        decorations[decoIter->GetSingleWordInOperand(1u)].push_back(*decoIter);
    } else if ((decoIter->opcode() == SpvOpDecorate ||
                decoIter->opcode() == SpvOpDecorateId) &&
               decoIter->GetSingleWordInOperand(1u) != SpvDecorationLinkageAttributes &&
               decoIter->GetSingleWordInOperand(1u) != SpvDecorationFuncParamAttr) { // FuncParamAttr are always taken from the definition anyway
      auto iter = std::find_if(idsToLookFor.cbegin(), idsToLookFor.cend(), [&decoIter](const std::pair<SpvId, uint32_t>& p) {
        return p.first == decoIter->GetSingleWordInOperand(0u);
      });
      if (iter != idsToLookFor.cend())
        decorations[iter->second].push_back(*decoIter);
    }
  } while (decoIter != module->annotation_begin());

  return decorations;
}

// Remove the whole instruction for SpvOpDecorate, SpvDecorateId and
// SpvMemberDecorate. For group decorations, juste remove the ID (and its
// structure index if present) from the list.
static void removeDecorationsFor(SpvId id, std::unique_ptr<Module>& module) {
  for (auto i = module->annotation_begin();
      i != module->annotation_end();) {
    if ((i->opcode() == SpvOpDecorate || i->opcode() == SpvOpDecorateId ||
        i->opcode() == SpvOpMemberDecorate) && i->GetSingleWordInOperand(0u) == id)
      i = i.Erase();
    else if (i->opcode() == SpvOpGroupDecorate) {
      std::vector<Operand> operands;
      operands.reserve(i->NumInOperands());
      for (uint32_t j = 2u; j < i->NumInOperands(); ++j)
        if (i->GetSingleWordInOperand(j) != id)
          operands.push_back(i->GetOperand(j));
      *i = Instruction(i->opcode(), 0u, 0u, operands);
    } else if (i->opcode() == SpvOpGroupMemberDecorate) {
      std::vector<Operand> operands;
      operands.reserve(i->NumInOperands());
      for (uint32_t j = 2u; j < i->NumInOperands(); j += 2u) {
        if (i->GetSingleWordInOperand(j) != id) {
          operands.push_back(i->GetOperand(j));
          operands.push_back(i->GetOperand(j + 1u));
        }
      }
      *i = Instruction(i->opcode(), 0u, 0u, operands);
    } else
      ++i;
  }
}

static bool areDecorationsSimilar(const Instruction& a, const Instruction& b, const std::unordered_map<SpvId, Instruction>& constants) {
  const auto decorateIdToDecorate = [&constants](const Instruction& inst) {
    std::vector<Operand> operands;
    operands.reserve(inst.NumInOperands());
    for (uint32_t i = 2u; i < inst.NumInOperands(); ++i) {
      const auto& j = constants.find(inst.GetSingleWordInOperand(i));
      if (j == constants.end())
        return Instruction();
      const auto operand = j->second.GetOperand(0u);
      operands.emplace_back(operand.type, operand.words);
    }
    return Instruction(SpvOpDecorate, 0u, 0u, operands);
  };
  Instruction tmpA = (a.opcode() == SpvOpDecorateId) ? decorateIdToDecorate(a) : a;
  Instruction tmpB = (b.opcode() == SpvOpDecorateId) ? decorateIdToDecorate(b) : b;

  if (tmpA.opcode() != tmpB.opcode() || tmpA.NumInOperands() != tmpB.NumInOperands() ||
      tmpA.opcode() == SpvOpNop || tmpB.opcode() == SpvOpNop)
    return false;

  for (uint32_t i = (tmpA.opcode() == SpvOpDecorate) ? 1u : 2u; i < tmpA.NumInOperands(); ++i)
    if (tmpA.GetInOperand(i) != tmpB.GetInOperand(i))
      return false;

  return true;
}

static bool haveIdsSimilarDecorations(SpvId id1, const std::unique_ptr<Module>& module1, SpvId id2, const std::unique_ptr<Module>& module2) {
  const IdDecorationsList decorationsList1 = getDecorationsForId(id1, module1);
  const IdDecorationsList decorationsList2 = getDecorationsForId(id2, module2);

  if (decorationsList1.size() != decorationsList2.size())
    return false;

  // Grab all SpvOpConstant: those should be the only constant instructions
  // used in SpvOpGroupDecorateId besides SpvOpSpecConstant, however there is
  // no way to decide whether two decorations are the same if we rely on value
  // that might change due to specialisation (which should occur before linking
  // anyway?)
  std::unordered_map<SpvId, Instruction> constants;
  for (const auto& i : module1->types_values())
    if (i.opcode() == SpvOpConstant)
      constants[i.result_id()] = i;
  for (const auto& i : module2->types_values())
    if (i.opcode() == SpvOpConstant)
      constants[i.result_id()] = i;

  for (const auto& i : decorationsList1) {
    const auto j = decorationsList2.find(i.first);
    if (j == decorationsList2.end() || i.second.size() != j->second.size())
      return false;

    for (size_t k = 0u; k < i.second.size(); ++k)
      if (!areDecorationsSimilar(i.second[k], j->second[k], constants))
        return false;
  }

  return true;
}

static spv_result_t MergeModules(
    const std::vector<std::unique_ptr<Module>>& inModules,
    libspirv::AssemblyGrammar& grammar, MessageConsumer& consumer,
    std::unique_ptr<Module>& linkedModule) {
  spv_position_t position = {};

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
    if (memoryModelInsn == nullptr)
      break;

    uint32_t addressingModel = memoryModelInsn->GetSingleWordOperand(0u);
    uint32_t memoryModel = memoryModelInsn->GetSingleWordOperand(1u);
    for (const auto& module : inModules) {
      memoryModelInsn = module->GetMemoryModel();
      if (memoryModelInsn == nullptr)
        continue;

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
  } while(false);

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
      func->SetParent(linkedModule.get());
      linkedModule->AddFunction(std::move(func));
    }
  }

  return SPV_SUCCESS;
}

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

  linked_binary.clear();
  if (binaries.empty())
    return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                      SPV_ERROR_INVALID_BINARY)
           << "No modules were given.";

  std::vector<std::unique_ptr<Module>> modules;
  modules.reserve(binaries.size());
  for (const auto& mod : binaries) {
    std::unique_ptr<Module> module =
        BuildModule(impl_->context->target_env, impl_->context->consumer,
                    mod.data(), mod.size());
    if (module == nullptr)
      return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "Failed to build a module out of " << modules.size()
             << ".";
    modules.push_back(std::move(module));
  }

  PassManager manager;
  manager.SetMessageConsumer(impl_->context->consumer);

  // Phase 1: Shift the IDs used in each binary so that they occupy a disjoint
  //          range from the other binaries, and compute the new ID bound.
  uint32_t id_bound = modules[0]->IdBound() - 1u;
  for (auto i = modules.begin() + 1; i != modules.end(); ++i) {
    Module* module = i->get();
    module->ForEachInst([&id_bound](Instruction* insn) {
      insn->ForEachId([&id_bound](uint32_t* id) { *id += id_bound; });
    });
    id_bound += module->IdBound() - 1u;
    if (id_bound > 0x3FFFFF)
      return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                        SPV_ERROR_INVALID_ID)
             << "The limit of IDs, 4194303, was exceeded:"
             << " " << id_bound << " is the current ID bound.";
  }
  ++id_bound;
  if (id_bound > 0x3FFFFF)
    return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                      SPV_ERROR_INVALID_ID)
           << "The limit of IDs, 4194303, was exceeded:"
           << " " << id_bound << " is the current ID bound.";

  // Phase 2: Generate the header
  //
  // TODO(pierremoreau): What to do when binaries use different versions of
  //                     SPIR-V? For now, use the max of all versions found in
  //                     the input modules.
  uint32_t version = 0u;
  for (const auto& module : modules)
    version = std::max(version, module->version());

  ir::ModuleHeader header;
  header.magic_number = SpvMagicNumber;
  header.version = version;
  header.generator = 17u;
  header.bound = id_bound;
  header.reserved = 0u;
  auto linkedModule = MakeUnique<Module>();
  linkedModule->SetHeader(header);

  // Phase 3: Merge all the binaries into a single one.
  libspirv::AssemblyGrammar grammar(impl_->context);
  spv_result_t res =
      MergeModules(modules, grammar, impl_->context->consumer, linkedModule);
  if (res != SPV_SUCCESS) return res;

  DefUseManager defUseManager(impl_->context->consumer, linkedModule.get());

  // Extract linking information
  struct LinkingInfo {
    SpvId id;
    SpvId typeId;
    std::string name;
    std::vector<SpvId> parametersIds;
  };
  std::vector<LinkingInfo> imports;
  std::unordered_map<std::string, std::vector<LinkingInfo>> exports;
  // Figure out the imports and exports
  for (const auto& j : linkedModule->annotations()) {
    if (j.opcode() == SpvOpDecorate && j.GetSingleWordInOperand(1u) == SpvDecorationLinkageAttributes) {
      uint32_t type = j.GetSingleWordInOperand(3u);
      SpvId id = j.GetSingleWordInOperand(0u);
      LinkingInfo data;
      data.name = reinterpret_cast<const char*>(j.GetInOperand(2u).words.data());
      data.id = id;
      data.typeId = 0u;
      const Instruction* defInst = defUseManager.GetDef(id);
      if (defInst == nullptr)
        return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                          SPV_ERROR_INVALID_BINARY)
                << "ID " << id << " is never defined:\n";
      if (defInst->opcode() == SpvOpVariable) {
        data.typeId = defInst->type_id();
      } else if (defInst->opcode() == SpvOpFunction) {
        data.typeId = defInst->GetSingleWordInOperand(1u);
        for (const auto& func : *linkedModule) {
          if (func.result_id() != id)
            continue;
          func.ForEachParam([&data](const Instruction* inst) {
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
  std::vector<std::pair<LinkingInfo, LinkingInfo>> linkingsToDo;
  for (const auto& import : imports) {
    std::vector<LinkingInfo> possibleExports;
    const auto& exp = exports.find(import.name);
    if (exp != exports.end())
      possibleExports = exp->second;
    if (possibleExports.empty())
      return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                        SPV_ERROR_INVALID_BINARY)
              << "No export linkage was found for \"" << import.name << "\".";
    else if (possibleExports.size() > 1u)
      return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                        SPV_ERROR_INVALID_BINARY)
              << "Too many export linkages, " << possibleExports.size()
              << ", were found for \"" << import.name << "\".";

    linkingsToDo.emplace_back(import, possibleExports.front());
  }

  // Ensure the import and export types are similar
  for (const auto&i : linkingsToDo) {
    if (!RemoveDuplicatesPass::AreTypesEqual(*defUseManager.GetDef(i.first.typeId), *defUseManager.GetDef(i.second.typeId), defUseManager))
      return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "Type mismatch between imported variable/function %" << i.first.id
             << " and exported variable/function %" << i.second.id << ".";
  }


  // Ensure the import and export decorations are similar
  for (const auto& i : linkingsToDo) {
    if (!haveIdsSimilarDecorations(i.first.id, linkedModule,
                                   i.second.id, linkedModule))
        return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                          SPV_ERROR_INVALID_BINARY)
               << "Decorations mismatch between imported variable/function %" << i.first.id
               << " and exported variable/function %" << i.second.id << ".";
    if (!haveIdsSimilarDecorations(i.first.typeId, linkedModule,
                                   i.second.typeId, linkedModule))
        return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                          SPV_ERROR_INVALID_BINARY)
               << "Decorations mismatch between the type of imported variable/function %" << i.first.id
               << " and the type of exported variable/function %" << i.second.id << ".";
    for (uint32_t j = 0u; j < i.first.parametersIds.size(); ++j)
      if (!haveIdsSimilarDecorations(i.first.parametersIds[j], linkedModule,
                                     i.second.parametersIds[j], linkedModule))
          return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                            SPV_ERROR_INVALID_BINARY)
                 << "Decorations mismatch between imported function %" << i.first.id << "'s"
                 << " and exported function %" << i.second.id << "'s " << j << "th parameter.";
  }

  // TODO(pierremoreau): Remove import types
  // Similar to the testing: traverse bottom-top
  // But make sure to store the ID of removed types, to do the replacement
  // later.

  // Remove import decorations
  for (const auto& i : linkingsToDo) {
    removeDecorationsFor(i.first.id, linkedModule);
    removeDecorationsFor(i.first.typeId, linkedModule);
    for (const auto j : i.first.parametersIds)
      removeDecorationsFor(j, linkedModule);
  }

  // Remove prototypes of imported functions
  for (const auto& i : linkingsToDo) {
    for (auto j = linkedModule->begin();
        j != linkedModule->end();)
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
      auto id_iter = std::find_if(linkingsToDo.begin(), linkingsToDo.end(), [id](const std::pair<LinkingInfo, LinkingInfo>& pair) {
        return pair.second.id == *id;
      });
      if (id_iter != linkingsToDo.end())
        *id = id_iter->first.id;
    });
  });

  // TODO(pierremoreau) Rematch import types/decorations to export types/decorations

  // Remove duplicates
  manager.AddPass<RemoveDuplicatesPass>();

  // Remove import linkage attributes
  for (auto i = linkedModule->annotation_begin();
       i != linkedModule->annotation_end();) {
    if (i->opcode() != SpvOpDecorate ||
        i->GetSingleWordOperand(1u) != SpvDecorationLinkageAttributes ||
        i->GetSingleWordOperand(3u) != SpvLinkageTypeImport) {
      ++i;
      continue;
    }
    i = i.Erase();
  }

  // Remove export linkage attributes and Linkage capability if making an
  // executable
  if (!options.GetCreateLibrary()) {
    for (auto i = linkedModule->annotation_begin();
         i != linkedModule->annotation_end();) {
      if (i->opcode() != SpvOpDecorate ||
          i->GetSingleWordOperand(1u) != SpvDecorationLinkageAttributes ||
          i->GetSingleWordOperand(3u) != SpvLinkageTypeExport) {
        ++i;
        continue;
      }
      i = i.Erase();
    }

    for (auto i = linkedModule->capability_begin();
        i != linkedModule->capability_end();) {
      if (i->GetSingleWordInOperand(0u) != SpvCapabilityLinkage) {
        ++i;
        continue;
      }
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

}  // namespace spvtool
