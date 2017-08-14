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

using namespace ir;
using namespace opt;

struct TypeData {
  size_t moduleIndex;
  Instruction i;
};

static bool areTypesSimilar(
    const Instruction& a, const Instruction& b,
    std::unordered_map<SpvId, TypeData>& typesMap) {
  if (a.opcode() != b.opcode()) return false;

  switch (a.opcode()) {
    case SpvOpTypeVoid:
    case SpvOpTypeBool:
    case SpvOpTypeSampler:
    case SpvOpTypeEvent:
    case SpvOpTypeDeviceEvent:
    case SpvOpTypeReserveId:
    case SpvOpTypeQueue:
    case SpvOpTypePipeStorage:
    case SpvOpTypeNamedBarrier:
      return true;
    case SpvOpTypeInt:
      return a.GetSingleWordInOperand(0u) == b.GetSingleWordInOperand(0u) &&
             a.GetSingleWordInOperand(1u) == b.GetSingleWordInOperand(1u);
    case SpvOpTypeFloat:
    case SpvOpTypePipe:
    case SpvOpTypeForwardPointer:
      return a.GetSingleWordInOperand(0u) == b.GetSingleWordInOperand(0u);
    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
      return areTypesSimilar(typesMap[a.GetSingleWordInOperand(0u)].i,
                             typesMap[b.GetSingleWordInOperand(0u)].i,
                             typesMap) &&
             a.GetSingleWordInOperand(1u) == b.GetSingleWordInOperand(1u);
    case SpvOpTypeImage:
      return areTypesSimilar(typesMap[a.GetSingleWordInOperand(0u)].i,
                             typesMap[b.GetSingleWordInOperand(0u)].i,
                             typesMap) &&
             a.GetSingleWordInOperand(1u) == b.GetSingleWordInOperand(1u) &&
             a.GetSingleWordInOperand(2u) == b.GetSingleWordInOperand(2u) &&
             a.GetSingleWordInOperand(3u) == b.GetSingleWordInOperand(3u) &&
             a.GetSingleWordInOperand(4u) == b.GetSingleWordInOperand(4u) &&
             a.GetSingleWordInOperand(5u) == b.GetSingleWordInOperand(5u) &&
             a.GetSingleWordInOperand(6u) == b.GetSingleWordInOperand(6u) &&
             a.NumOperands() == b.NumOperands() &&
             (a.NumInOperands() == 8u ||
              a.GetSingleWordInOperand(7u) == b.GetSingleWordInOperand(7u));
    case SpvOpTypeSampledImage:
    case SpvOpTypeRuntimeArray:
      return areTypesSimilar(typesMap[a.GetSingleWordInOperand(0u)].i,
                             typesMap[b.GetSingleWordInOperand(0u)].i, typesMap);
    case SpvOpTypeArray:
      return areTypesSimilar(typesMap[a.GetSingleWordInOperand(0u)].i,
                             typesMap[b.GetSingleWordInOperand(0u)].i,
                             typesMap) &&
             areTypesSimilar(typesMap[a.GetSingleWordInOperand(1u)].i,
                             typesMap[b.GetSingleWordInOperand(1u)].i, typesMap);
    case SpvOpTypeStruct:
    case SpvOpTypeFunction: {
      bool res = a.NumInOperands() == b.NumInOperands();
      for (uint32_t i = 0u; i < a.NumInOperands() && res; ++i)
        res &= areTypesSimilar(typesMap[a.GetSingleWordInOperand(i)].i,
                               typesMap[b.GetSingleWordInOperand(i)].i, typesMap);
      return res;
    }
    case SpvOpTypeOpaque:
      return std::strcmp(
                 reinterpret_cast<const char*>(a.GetInOperand(0u).words.data()),
                 reinterpret_cast<const char*>(
                     b.GetInOperand(0u).words.data())) == 0;
    case SpvOpTypePointer:
      return a.GetSingleWordInOperand(0u) == b.GetSingleWordInOperand(0u) &&
             areTypesSimilar(typesMap[a.GetSingleWordInOperand(1u)].i,
                             typesMap[b.GetSingleWordInOperand(1u)].i, typesMap);
    default:
      return false;
  }
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
      std::unique_ptr<Function> func = MakeUnique<Function>(i);
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

  // Extract linking information
  struct LinkingData {
    size_t moduleIndex;
    SpvId id;
    SpvId typeId;
    const char* name;
  };
  std::vector<std::vector<LinkingData>> modulesImports(modules.size());
  std::vector<std::unordered_map<std::string, LinkingData>> modulesExports(modules.size());
  size_t moduleIndex = 0u;
  for (auto& i : modules) {
    // Figure out the imports and exports
    for (auto& j : i->annotations()) {
      if (j.opcode() == SpvOpDecorate && j.GetSingleWordInOperand(1u) == SpvDecorationLinkageAttributes) {
        uint32_t type = j.GetSingleWordInOperand(3u);
        SpvId id = j.GetSingleWordInOperand(0u);
        LinkingData data;
        data.moduleIndex = moduleIndex;
        data.name = reinterpret_cast<const char*>(j.GetInOperand(2u).words.data());
        data.id = id;
        data.typeId = 0u;
        if (type == SpvLinkageTypeImport)
          modulesImports[moduleIndex].push_back(data);
        else if (type == SpvLinkageTypeExport)
          modulesExports[moduleIndex].emplace(data.name, data);
      }
      // TODO(pierremoreau): Can an SpvOpDecorateId be used for
      //                     LinkageAttributes decorations?
    }
    ++moduleIndex;
  }

  //Find the import/export pairs
  std::vector<std::pair<LinkingData, LinkingData>> linkingsToDo;
  for (const auto& i : modulesImports) {
    for (const auto& j : i) {
      std::vector<LinkingData> possibleExports;
      for (const auto& k : modulesExports) {
        const auto& l = k.find(j.name);
        if (l != k.end())
          possibleExports.push_back(l->second);
      }
      if (possibleExports.empty())
        return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                          SPV_ERROR_INVALID_BINARY)
               << "No export linkage was found for \"" << j.name << "\".";
      else if (possibleExports.size() > 1u)
        return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                          SPV_ERROR_INVALID_BINARY)
               << "Too many export linkages, " << possibleExports.size()
               << ", were found for \"" << j.name << "\".";

      linkingsToDo.emplace_back(j, possibleExports.front());
    }
  }

  // Grab all types used by imports and exports
  std::unordered_map<size_t, std::unordered_set<SpvId>> typesToSearchFor;
  for (auto& i : linkingsToDo) {
    for (const auto& j : modules[i.first.moduleIndex]->types_values()) {
      if (j.result_id() == i.first.id) {
        assert(j.opcode() == SpvOpVariable);
        typesToSearchFor[i.first.moduleIndex].emplace(j.type_id());
        i.first.typeId = j.type_id();
        break;
      }
    }
    if (i.first.typeId == 0u) {
      for (const auto& j : *modules[i.first.moduleIndex].get()) {
        if (j.result_id() == i.first.id) {
          typesToSearchFor[i.first.moduleIndex].emplace(j.DefInst().GetSingleWordInOperand(1u));
          i.first.typeId = j.DefInst().GetSingleWordInOperand(1u);
          break;
        }
      }
    }
    assert(i.first.typeId != 0u);
    for (const auto& j : modules[i.second.moduleIndex]->types_values()) {
      if (j.result_id() == i.second.id) {
        assert(j.opcode() == SpvOpVariable);
        typesToSearchFor[i.second.moduleIndex].emplace(j.type_id());
        i.second.typeId = j.type_id();
        break;
      }
    }
    if (i.second.typeId == 0u) {
      for (const auto& j : *modules[i.second.moduleIndex].get()) {
        if (j.result_id() == i.second.id) {
          typesToSearchFor[i.second.moduleIndex].emplace(j.DefInst().GetSingleWordInOperand(1u));
          i.second.typeId = j.DefInst().GetSingleWordInOperand(1u);
          break;
        }
      }
    }
    assert(i.second.typeId != 0u);
  }
  std::unordered_map<SpvId, TypeData> types;
  for (auto& i : typesToSearchFor) {
    if (modules[i.first]->types_values().empty())
      continue;

    auto typesIter = modules[i.first]->types_values_end();
    do {
      --typesIter;
      if (i.second.find(typesIter->result_id()) != i.second.end()) {
        types.emplace(typesIter->result_id(), TypeData{ i.first, *typesIter });
        switch (typesIter->opcode()) {
          case SpvOpTypeVector:
          case SpvOpTypeMatrix:
          case SpvOpTypeImage:
          case SpvOpTypeSampledImage:
          case SpvOpTypeRuntimeArray:
            i.second.emplace(typesIter->GetSingleWordInOperand(0u));
            break;
          case SpvOpTypeArray:
            i.second.emplace(typesIter->GetSingleWordInOperand(0u));
            i.second.emplace(typesIter->GetSingleWordInOperand(1u));
            break;
          case SpvOpTypeStruct:
          case SpvOpTypeFunction:
            for (uint32_t j = 0u; j < typesIter->NumInOperands(); ++j)
              i.second.emplace(typesIter->GetSingleWordInOperand(j));
            break;
          case SpvOpTypePointer:
            i.second.emplace(typesIter->GetSingleWordInOperand(1u));
            break;
        }
      }
    } while (typesIter != modules[i.first]->types_values_begin());
  }

  // Ensure the import and export types are similar
  for (const auto&i : linkingsToDo) {
    auto importTypeIter = types.find(i.first.typeId);
    assert(importTypeIter != types.end());
    auto exportTypeIter = types.find(i.second.typeId);
    assert(exportTypeIter != types.end());

    if (!areTypesSimilar(importTypeIter->second.i, exportTypeIter->second.i, types))
      return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "Type mismatch between imported variable/function %" << i.first.id
             << " and exported variable/function %" << i.second.id << ".";
  }

  // Phase 2: Merge all the binaries into a single one.
  auto linkedModule = MakeUnique<Module>();
  libspirv::AssemblyGrammar grammar(impl_->context);
  spv_result_t res =
      MergeModules(modules, grammar, impl_->context->consumer, linkedModule);
  if (res != SPV_SUCCESS) return res;

  // Phase 3: Generate the linkage table.
  std::unordered_map<SpvId, std::string> imports;
  std::unordered_map<std::string, SpvId> exports;
  position.index = 0u;
  for (const auto& insn : linkedModule->annotations()) {
    position.index += (insn.result_id() != 0u) + insn.NumOperandWords();
    if (insn.opcode() != SpvOpDecorate) continue;
    if (insn.GetSingleWordOperand(1u) != SpvDecorationLinkageAttributes)
      continue;
    uint32_t linkage = insn.GetSingleWordOperand(insn.NumOperands() - 1u);
    if (linkage == SpvLinkageTypeImport)
      imports.emplace(
          insn.GetSingleWordOperand(0u),
          reinterpret_cast<const char*>(insn.GetOperand(2u).words.data()));
    else if (linkage == SpvLinkageTypeExport)
      exports.emplace(
          reinterpret_cast<const char*>(insn.GetOperand(2u).words.data()),
          insn.GetSingleWordOperand(0u));
    else
      return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "Invalid linkage type found.";
  }

  std::unordered_map<SpvId, SpvId> linking_table;
  linking_table.reserve(imports.size());
  for (const auto& i : imports) {
    auto j = exports.find(i.second);
    if (j == exports.end())
      return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "No export linkage was found for \"" << i.second << "\".";
    linking_table.emplace(i.first, j->second);
  }

  // Remove prototypes of imported functions
  for (auto i = linkedModule->begin(); i != linkedModule->end();) {
    const auto function_id = i->DefInst().result_id();
    i = (imports.find(function_id) != imports.end()) ? i.Erase() : ++i;
  }

  // Remove declarations of imported global variables
  for (auto i = linkedModule->types_values_begin();
       i != linkedModule->types_values_end();) {
    if (i->opcode() == SpvOpVariable) {
      const auto variable_id = i->GetSingleWordOperand(1u);
      i = (imports.find(variable_id) != imports.end()) ? i.Erase() : ++i;
    } else {
      ++i;
    }
  }

  // Remove debug instructions for imported declarations
  for (auto i = linkedModule->debug1_begin();
       i != linkedModule->debug1_end();) {
    bool should_remove = false;
    i->ForEachInId([&imports, &should_remove](const uint32_t* id) {
      should_remove |= imports.find(*id) != imports.end();
    });
    i = (should_remove) ? i.Erase() : ++i;
  }
  for (auto i = linkedModule->debug2_begin();
       i != linkedModule->debug2_end();) {
    bool should_remove = false;
    i->ForEachInId([&imports, &should_remove](const uint32_t* id) {
      should_remove |= imports.find(*id) != imports.end();
    });
    i = (should_remove) ? i.Erase() : ++i;
  }

  linkedModule->ForEachInst([&linking_table](Instruction* insn) {
    insn->ForEachId([&linking_table](uint32_t* id) {
      auto id_iter = linking_table.find(*id);
      if (id_iter != linking_table.end()) *id = id_iter->second;
    });
  });

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
  manager.AddPass<CompactIdsPass>();

  // Phase 5: Generate the header and output the linked module.

  // TODO(pierremoreau): What to do when binaries use different versions of
  //                     SPIR-V? For now, use the max of all versions found in
  //                     the input modules.
  uint32_t version = 0u;
  for (const auto& module : modules)
    version = std::max(version, module->version());

  ModuleHeader header;
  header.magic_number = SpvMagicNumber;
  header.version = version;
  // TODO(pierremoreau): Should we reserve a vendor ID?
  header.generator = 0u;
  header.bound = id_bound;
  header.reserved = 0u;
  linkedModule->SetHeader(header);

  // Phase 6: Run all accumulated optimisation passes.
  manager.Run(linkedModule.get());

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
