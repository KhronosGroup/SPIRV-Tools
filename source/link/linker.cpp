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

#include "spirv/1.2/spirv.hpp11"

#include <cstring>
#include <cstdio>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "diagnostic.h"
#include "opt/build_module.h"
#include "opt/ir_loader.h"
#include "opt/make_unique.h"
#include "spirv_target_env.h"
#include "spirv-tools/libspirv.hpp"


namespace spvtools {

using namespace ir;

static spv_result_t MergeModules(const std::vector<std::unique_ptr<Module>>& inModules,
                                 MessageConsumer& consumer,
                                std::unique_ptr<Module>& linkedModule) {
  for (const auto& module : inModules)
    for (const auto& insn : module->capabilities())
      linkedModule->AddCapability(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->extensions())
      linkedModule->AddExtension(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->ext_inst_imports())
      linkedModule->AddExtInstImport(MakeUnique<Instruction>(insn));

  // TODO(pierremoreau): error out if the |inModules| use different memory
  //                     models and/or addressing models.
  for (const auto& module : inModules)
    linkedModule->SetMemoryModel(MakeUnique<Instruction>(*module->GetMemoryModel()));

  // TODO(pierremoreau): error out if there are duplicate entry point names
  for (const auto& module : inModules)
    for (const auto& insn : module->entry_points())
      linkedModule->AddEntryPoint(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->execution_modes())
      linkedModule->AddExecutionMode(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->debugs())
      linkedModule->AddDebugInst(MakeUnique<Instruction>(insn));

  for (const auto& module : inModules)
    for (const auto& insn : module->annotations())
      linkedModule->AddAnnotationInst(MakeUnique<Instruction>(insn));

  // TODO(pierremoreau): error out if there are duplicate global variable names
  uint32_t num_global_values = 0u;
  for (const auto& module : inModules) {
    for (const auto& insn : module->types_values()) {
      linkedModule->AddType(MakeUnique<Instruction>(insn));
      num_global_values += insn.opcode() == SpvOpVariable;
    }
  }
  spv_position_t position = {};
  if (num_global_values > 0xFFFF)
    return libspirv::DiagnosticStream(position, consumer, SPV_ERROR_INTERNAL)
           << "The limit of global values was exceeded.";

  // TODO(pierremoreau): error out if there are duplicate function signatures
  // Process functions and their basic blocks
  for (const auto& module : inModules) {
    for (const auto& i : *module) {
      std::unique_ptr<Function> func = MakeUnique<Function>(Function(MakeUnique<Instruction>(i.DefInst())));
      func->SetParent(linkedModule.get());
      i.ForEachParam([&func](const Instruction* insn){
          func->AddParameter(MakeUnique<Instruction>(*insn));
      }, true);
      // TODO(pierremoreau): convince the compiler to use cbegin()/cend()
      //                     instead of begin()/end()
      for (auto j = i.cbegin(); j != i.cend(); ++j) {
        std::unique_ptr<BasicBlock> block = MakeUnique<BasicBlock>(BasicBlock(MakeUnique<Instruction>(j->GetLabelInst())));
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
                               uint32_t options) const {
  spv_position_t position = {};

  std::vector<std::unique_ptr<Module>> modules;
  modules.reserve(binaries.size());
  for (const auto& mod : binaries) {
    std::unique_ptr<Module> module = BuildModule(impl_->context->target_env, impl_->context->consumer, mod.data(), mod.size());
    if (module == nullptr)
      return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "Failed to build a module out of binary " << modules.size() << ".";
    modules.push_back(std::move(module));
  }


  // Phase 1: Shift the IDs used in each binary so that they occupy a disjoint
  //          range from the other binaries, and compute the new ID bound.
  uint32_t id_bound = 0u;
  for (auto& module : modules) {
    module->ForEachInst([&id_bound](Instruction* insn){
      for (auto& o : *insn)
        if (spvIsIdType(o.type)) o.words[0] += id_bound;
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
  spv_result_t res = MergeModules(modules, impl_->context->consumer, linkedModule);
  if (res != SPV_SUCCESS)
    return res;


  // Phase 3: Generate the linkage table.
  std::unordered_map<spv::Id, std::string> imports;
  std::unordered_map<std::string, spv::Id> exports;
  position.index = 0u;
  for (const auto& insn : linkedModule->annotations()) {
    position.index += (insn.result_id() != 0u) + insn.NumOperandWords();
    if (insn.opcode() != SpvOpDecorate)
      continue;
    if (static_cast<spv::Decoration>(insn.GetSingleWordOperand(1u)) != spv::Decoration::LinkageAttributes)
      continue;
    spv::LinkageType linkage = static_cast<spv::LinkageType>(insn.GetSingleWordOperand(insn.NumOperands() - 1u));
    if (linkage == spv::LinkageType::Import)
      imports.emplace(insn.GetSingleWordOperand(0u), reinterpret_cast<const char*>(insn.GetOperand(2u).words.data()));
    else if (linkage == spv::LinkageType::Export)
      exports.emplace(reinterpret_cast<const char*>(insn.GetOperand(2u).words.data()), insn.GetSingleWordOperand(0u));
    else
      return libspirv::DiagnosticStream(position, impl_->context->consumer,
                                        SPV_ERROR_INVALID_BINARY)
             << "Invalid linkage type found.";
  }

  std::unordered_map<spv::Id, spv::Id> linking_table;
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
  for (auto i = linkedModule->types_values_begin(); i != linkedModule->types_values_end();) {
    if (i->opcode() == SpvOpVariable) {
      const auto variable_id = i->GetSingleWordOperand(1u);
      i = (imports.find(variable_id) != imports.end()) ? i.Erase() : ++i;
    } else {
      ++i;
    }
  }

  // Remove debug instructions for imported declarations
  for (auto i = linkedModule->debug_begin(); i != linkedModule->debug_end();) {
    bool should_remove = false;
    i->ForEachInId([&imports,&should_remove](const uint32_t* id){
      should_remove |= imports.find(*id) != imports.end();
    });
    i = (should_remove) ? i.Erase() : ++i;
  }

  // TODO(pierremoreau): This needs to be ran on result_id and type_id as well
  linkedModule->ForEachInst([&linking_table](Instruction* insn){
    for (auto o = insn->begin(); o != insn->end(); ++o) {
      spv::Id& id = o->words[0];
      if (!spvIsIdType(o->type))
        continue;
      auto id_iter = linking_table.find(id);
      if (id_iter != linking_table.end())
        id = id_iter->second;
    }
  });

  // Remove duplicate capabilities
  std::unordered_set<uint32_t> capabilities;
  for (auto i = linkedModule->capability_begin(); i != linkedModule->capability_end();) {
    auto insertRes = capabilities.insert(i->GetSingleWordOperand(0u));
    i = (insertRes.second) ? ++i : i.Erase();
  }

  // Remove import linkage attributes
  for (auto i = linkedModule->annotation_begin(); i != linkedModule->annotation_end();) {
    if (i->opcode() != SpvOpDecorate ||
        static_cast<spv::Decoration>(i->GetSingleWordOperand(1u)) != spv::Decoration::LinkageAttributes ||
        static_cast<spv::LinkageType>(i->GetSingleWordOperand(3u)) != spv::LinkageType::Import) {
      ++i;
      continue;
    }
    i = i.Erase();
  }

  // Remove export linkage attributes if making an executable
  if (!(options & SPV_LINKER_OPTION_CREATE_LIBRARY)) {
    for (auto i = linkedModule->annotation_begin(); i != linkedModule->annotation_end();) {
      if (i->opcode() != SpvOpDecorate ||
          static_cast<spv::Decoration>(i->GetSingleWordOperand(1u)) != spv::Decoration::LinkageAttributes ||
          static_cast<spv::LinkageType>(i->GetSingleWordOperand(3u)) != spv::LinkageType::Export) {
        ++i;
        continue;
      }
      i = i.Erase();
    }
  }

  // TODO(pierremoreau): Run an optimisation pass to compact all IDs.


  // Phase 5: Generate the header and output the linked module.
  ModuleHeader header;
  header.magic_number = spv::MagicNumber;
  // TODO(pierremoreau): What to do when binaries use different versions of
  //                   SPIR-V?
  header.version = spv::Version;
  // TODO(pierremoreau): Should we reserve a vendor ID?
  header.generator = 0u;
  header.bound = id_bound;
  header.reserved = 0u;
  linkedModule->SetHeader(header);

  linkedModule->ToBinary(&linked_binary, true);

  return SPV_SUCCESS;
}

spv_result_t Linker::Link(const uint32_t* const* binaries, const size_t* binary_sizes,
                               size_t num_binaries, std::vector<uint32_t>& linked_binary,
                               uint32_t options) const {
  std::vector<std::vector<uint32_t>> binaries_array;
  binaries_array.reserve(num_binaries);
  for (size_t i = 0u; i < num_binaries; ++i)
    binaries_array.push_back({binaries[i], binaries[i] + binary_sizes[i]});

  return Link(binaries_array, linked_binary, options);
}

} // namespace spvtool
