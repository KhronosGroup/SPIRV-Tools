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
#include "opt/ir_loader.h"
#include "opt/make_unique.h"
#include "spirv-tools/libspirv.hpp"
#include "spirv_target_env.h"

namespace spvtools {

using namespace ir;

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

  {
    const Instruction* memoryModelInsn = inModules[0]->GetMemoryModel();
    uint32_t addressingModel = memoryModelInsn->GetSingleWordOperand(0u);
    uint32_t memoryModel = memoryModelInsn->GetSingleWordOperand(1u);
    for (const auto& module : inModules) {
      memoryModelInsn = module->GetMemoryModel();
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
    linkedModule->SetMemoryModel(MakeUnique<Instruction>(*memoryModelInsn));
  }

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

  uint32_t num_global_values = 0u;
  for (const auto& module : inModules) {
    for (const auto& insn : module->types_values()) {
      linkedModule->AddType(MakeUnique<Instruction>(insn));
      num_global_values += insn.opcode() == SpvOpVariable;
    }
  }
  if (num_global_values > 0xFFFF)
    return libspirv::DiagnosticStream(position, consumer, SPV_ERROR_INTERNAL)
           << "The limit of global values was exceeded.";

  // Process functions and their basic blocks
  for (const auto& module : inModules) {
    for (const auto& i : *module) {
      std::unique_ptr<Function> func =
          MakeUnique<Function>(Function(MakeUnique<Instruction>(i.DefInst())));
      func->SetParent(linkedModule.get());
      i.ForEachParam(
          [&func](const Instruction* insn) {
            func->AddParameter(MakeUnique<Instruction>(*insn));
          },
          true);
      // TODO(pierremoreau): convince the compiler to use cbegin()/cend()
      //                     instead of begin()/end()
      for (auto j = i.cbegin(); j != i.cend(); ++j) {
        std::unique_ptr<BasicBlock> block = MakeUnique<BasicBlock>(
            BasicBlock(MakeUnique<Instruction>(j->GetLabelInst())));
        block->SetParent(func.get());
        // TODO(pierremoreau): convince the compiler to use cbegin()/cend()
        //                     instead of begin()/end()
        for (auto k = j->cbegin(); k != j->cend(); ++k)
          block->AddInstruction(MakeUnique<Instruction>(*k));
        func->AddBasicBlock(std::move(block));
      }
      func->SetFunctionEnd(MakeUnique<Instruction>(i.function_end()));
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
             << "Failed to build a module out of binary " << modules.size()
             << ".";
    modules.push_back(std::move(module));
  }

  // Phase 1: Shift the IDs used in each binary so that they occupy a disjoint
  //          range from the other binaries, and compute the new ID bound.
  uint32_t id_bound = modules[0]->IdBound() - 1u;
  for (auto i = modules.begin() + 1; i != modules.end(); ++i) {
    Module* module = i->get();
    module->ForEachInst([&id_bound](Instruction* insn) {
      insn->ForEachInId([&id_bound](uint32_t* id) { *id += id_bound; });
      if (const uint32_t result_id = insn->result_id())
        insn->SetResultId(result_id + id_bound);
      if (const uint32_t result_type = insn->type_id())
        insn->SetResultType(result_type + id_bound);
    });
    id_bound += module->IdBound() - 1u;
    if (id_bound > 0x3FFFFF)
      return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                        SPV_ERROR_INVALID_ID)
             << "The limit of IDs was exceeded.";
  }
  ++id_bound;
  if (id_bound > 0x3FFFFF)
    return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                      SPV_ERROR_INVALID_ID)
           << "The limit of IDs was exceeded.";

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

  // Phase 4: Clean up remains of imported functions and global variables.

  // TODO(pierremoreau): Switch usage of the type of exported
  //                     functions/variables, to their imported counterpart.
  //
  //                     For example, if we have an imported variable:
  //
  //                       %impVar = OpVariable %impVarType Global
  //
  //                     an exported one:
  //
  //                       %expVar = OpVariable %expVarType Global 3.14f
  //
  //                     and a function which uses the imported variable:
  //
  //                       %func = OpFunction [...] { OpStore %impVar %obj }
  //
  //                     After linking, we have:
  //
  //                       %expVar = OpVariable %expVarType Global 3.14f
  //                       %func = OpFunction [...] { OpStore %expVar %obj }
  //
  //                     However, we now have a mismatch between %obj's type's
  //                     ID (the ID of the type pointed by %impVarType) and the
  //                     ID of the type %expVar is pointing to.

  // TODO(pierremoreau): Linked to the previous todo, imported and exported
  //                     functions/variables might have some common
  //                     decorations, or their types. The duplicates can be
  //                     removed.

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
    const auto link = [&linking_table](uint32_t* id) {
      auto id_iter = linking_table.find(*id);
      if (id_iter != linking_table.end()) *id = id_iter->second;
    };
    insn->ForEachInId(link);
    if (uint32_t result_id = insn->result_id()) {
      link(&result_id);
      insn->SetResultId(result_id);
    }
    if (uint32_t result_type = insn->type_id()) {
      link(&result_type);
      insn->SetResultType(result_type);
    }
  });

  // Remove duplicate capabilities
  std::unordered_set<uint32_t> capabilities;
  for (auto i = linkedModule->capability_begin();
       i != linkedModule->capability_end();) {
    auto insertRes = capabilities.insert(i->GetSingleWordOperand(0u));
    i = (insertRes.second) ? ++i : i.Erase();
  }

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

  // Remove export linkage attributes if making an executable
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
  }

  // TODO(pierremoreau): Run an optimisation pass to compact all IDs.

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
