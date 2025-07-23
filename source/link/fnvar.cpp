// Copyright 2025 The Khronos Group Inc.
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

#include "fnvar.h"

#include <sstream>

#include "source/opt/instruction.h"

namespace spvtools {

using opt::Function;
using opt::Instruction;
using opt::analysis::Type;

namespace {

bool ParseCsv(std::string source, std::vector<std::string> required_cols,
              std::stringstream& err_msg,
              std::vector<std::vector<std::string>>& result) {
  std::stringstream fn_variants_csv_stream(source);
  std::string line;
  std::vector<std::string> columns;
  constexpr char CSV_SEP = ',';
  bool first_line = true;

  while (std::getline(fn_variants_csv_stream, line, '\n')) {
    if (line.empty()) {
      continue;
    }

    std::vector<std::string> vals;
    std::string val;
    std::stringstream line_stream(line);
    auto* vec = first_line ? &columns : &vals;

    while (std::getline(line_stream, val, CSV_SEP)) {
      vec->push_back(val);
    }

    if (!line_stream && val.empty()) {
      vec->push_back("");
    }

    if (!first_line) {
      if (vals.size() != columns.size()) {
        err_msg << "Number of values does not match the number of columns. "
                   "Offending line:\n"
                << line;
        return false;
      }
      result.push_back(vals);
    }

    first_line = false;
  }

  // check if required columns match actual columns (ordering matters)

  if (columns.size() != required_cols.size()) {
    err_msg << "Invalid number of CSV columns: " << columns.size()
            << ", expected " << required_cols.size() << ".";
    return false;
  }

  for (size_t i = 0; i < columns.size(); ++i) {
    if (columns[i] != required_cols[i]) {
      err_msg << "Invalid name of column " << i + 1 << ". Expected '"
              << required_cols[i] << "', got '" << columns[i] << "'.";
      return false;
    }
  }

  return true;
}

// Annotate ID with ConditionalINTEL decoration
void DecorateConditional(IRContext* context, uint32_t id_to_decorate,
                         uint32_t spec_const_id) {
  auto decor_instr = Instruction(context, spv::Op::OpDecorate);
  decor_instr.AddOperand({SPV_OPERAND_TYPE_ID, {id_to_decorate}});
  decor_instr.AddOperand({SPV_OPERAND_TYPE_DECORATION,
                          {uint32_t(spv::Decoration::ConditionalINTEL)}});
  decor_instr.AddOperand({SPV_OPERAND_TYPE_ID, {spec_const_id}});
  context->module()->AddAnnotationInst(
      std::unique_ptr<Instruction>(decor_instr.Clone(context)));
}

// Find entry point corresponding to a function
Instruction* FindEntryPoint(const Instruction& fn_inst) {
  auto* mod = fn_inst.context()->module();
  for (auto& entry_point : mod->entry_points()) {
    const int ep_i =
        entry_point.opcode() == spv::Op::OpConditionalEntryPointINTEL ? 2 : 1;
    if (entry_point.GetOperand(ep_i).AsId() == fn_inst.result_id()) {
      return &entry_point;
    }
  }
  return nullptr;
}

// If the function has an entry point, convert it to a conditional one
void ConvertEPToConditional(Module* module, const Function& fn,
                            uint32_t spec_const_id) {
  for (const auto& ep_inst : module->entry_points()) {
    if (ep_inst.opcode() == spv::Op::OpEntryPoint) {
      auto* entry_point = FindEntryPoint(fn.DefInst());
      if (entry_point != nullptr) {
        std::vector<opt::Operand> old_operands;
        for (auto operand : *entry_point) {
          old_operands.push_back(operand);
        }
        entry_point->ToNop();
        entry_point->SetOpcode(spv::Op::OpConditionalEntryPointINTEL);
        entry_point->AddOperand({SPV_OPERAND_TYPE_ID, {spec_const_id}});
        for (auto old_operand : old_operands) {
          entry_point->AddOperand(old_operand);
        }
      }
    }
  }
}

uint32_t FindIdOfBoolType(const Module* const mod) {
  return mod->context()->get_type_mgr()->GetBoolTypeId();
}

uint32_t CombineIds(IRContext* const context, std::vector<uint32_t> ids,
                    spv::Op cmp_op) {
  if (ids.empty()) {
    return 0;
  } else if (ids.size() == 1) {
    return ids[0];
  } else {
    uint32_t bool_id = FindIdOfBoolType(context->module());
    assert(bool_id != 0);

    uint32_t prev_spec_const_id = *ids.begin();

    for (size_t i = 1; i < ids.size(); ++i) {
      const uint32_t id = ids[i];
      const uint32_t spec_const_op_id = context->TakeNextId();

      const auto inst = Instruction(
          context, spv::Op::OpSpecConstantOp, bool_id, spec_const_op_id,
          std::initializer_list<opt::Operand>{
              {SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER, {(uint32_t)(cmp_op)}},
              {SPV_OPERAND_TYPE_ID, {prev_spec_const_id}},
              {SPV_OPERAND_TYPE_ID, {id}}});
      context->module()->AddType(
          std::unique_ptr<Instruction>(inst.Clone(context)));

      prev_spec_const_id = spec_const_op_id;
    }

    return prev_spec_const_id;
  }
}

// Returns whether instruction can be shared between variant modules and
// combined using spec constants (such as conditional capabilities).
bool CanBeFnVarCombined(const Instruction* inst) {
  const spv::Op opcode = inst->opcode();

  if ((opcode != spv::Op::OpExtInstImport) &&
      (opcode != spv::Op::OpCapability) && (opcode != spv::Op::OpExtension) &&
      !spvOpcodeGeneratesType(opcode)) {
    return false;
  }

  if ((opcode == spv::Op::OpCapability) &&
      ((inst->GetOperand(0).words[0] ==
        static_cast<uint32_t>(spv::Capability::FunctionVariantsINTEL)) ||
       (inst->GetOperand(0).words[0] ==
        static_cast<uint32_t>(spv::Capability::SpecConditionalINTEL)))) {
    // Always enabled
    return false;
  }

  if ((opcode == spv::Op::OpExtension) &&
      (inst->GetOperand(0).AsString() == FNVAR_EXT_NAME)) {
    // Always enabled
    return false;
  }

  return true;
}

size_t HashInst(const Instruction* inst) {
  if (CanBeFnVarCombined(inst)) {
    if (spvOpcodeGeneratesType(inst->opcode())) {
      const Type* t =
          inst->context()->get_type_mgr()->GetType(inst->result_id());
      assert(t != nullptr);
      return t->HashValue();
    }

    if (inst->opcode() == spv::Op::OpExtension) {
      const auto name = inst->GetOperand(0).AsString();
      return std::hash<std::string>()(name);
    }

    if (inst->opcode() == spv::Op::OpCapability) {
      const auto cap = inst->GetOperand(0).words[0];
      return std::hash<uint32_t>()(cap);
    }

    if (inst->opcode() == spv::Op::OpExtInstImport) {
      const auto name = inst->GetOperand(1).AsString();
      return std::hash<std::string>()(name);
    }
  }

  assert(false && "Unsupported instruction hash");
  return std::hash<const Instruction*>()(inst);
}

std::string GetFnName(const Instruction& fn_inst) {
  // Check entry point
  const auto* ep_inst = FindEntryPoint(fn_inst);
  if (ep_inst != nullptr) {
    const int name_i =
        ep_inst->opcode() == spv::Op::OpConditionalEntryPointINTEL ? 3 : 2;
    return ep_inst->GetOperand(name_i).AsString();
  }

  // Check name of export linkage attribute decoration
  const auto* decor_mgr = fn_inst.context()->get_decoration_mgr();
  for (const auto* inst :
       decor_mgr->GetDecorationsFor(fn_inst.result_id(), true)) {
    const auto decoration = inst->GetOperand(1);
    if ((decoration.type == SPV_OPERAND_TYPE_DECORATION) &&
        (decoration.words.size() == 1) &&
        (decoration.words[0] ==
         static_cast<uint32_t>(spv::Decoration::LinkageAttributes))) {
      const auto linkage = inst->GetOperand(3);
      if ((linkage.type == SPV_OPERAND_TYPE_LINKAGE_TYPE) &&
          (linkage.words.size() == 1) &&
          (linkage.words[0] ==
           static_cast<uint32_t>(spv::LinkageType::Export))) {
        // decorates fn with LinkageAttribute and Export linkage type -> get the
        // name
        return inst->GetOperand(2).AsString();
      }
    }
  }

  return "";
}

uint32_t FindSpecConstByName(const Module* mod, std::string name) {
  for (const auto* const_inst : mod->context()->GetConstants()) {
    if (opt::IsSpecConstantInst(const_inst->opcode())) {
      const auto id = const_inst->result_id();
      for (const auto& name_inst : mod->debugs2()) {
        if ((name_inst.opcode() == spv::Op::OpName) &&
            (name_inst.GetOperand(0).AsId() == id) &&
            (name_inst.GetOperand(1).AsString() == name)) {
          return id;
        }
      }
    }
  }
  return 0;
}

uint32_t CombineVariants(const std::vector<Variant>& variants,
                         const std::vector<size_t> var_ids, IRContext* context,
                         std::map<std::vector<size_t>, uint32_t>& cache) {
  assert(var_ids.size() <= variants.size());
  uint32_t spec_const_comb_id = 0;
  if (var_ids.size() != variants.size()) {
    // if not used by all variants
    if (cache.find(var_ids) == cache.end()) {
      // cache variant combinations
      std::vector<uint32_t> spec_const_ids;
      for (const auto& var_id : var_ids) {
        const auto var_name = variants[var_id].GetName();
        const auto var_spec_id =
            FindSpecConstByName(context->module(), var_name);
        spec_const_ids.push_back(var_spec_id);
      }
      spec_const_comb_id =
          CombineIds(context, spec_const_ids, spv::Op::OpLogicalOr);
      assert(spec_const_comb_id != 0);
      cache.insert({var_ids, spec_const_comb_id});
    } else {
      spec_const_comb_id = cache[var_ids];
    }
  }
  return spec_const_comb_id;
}

}  // anonymous namespace

bool Variants::ProcessFnVar(const LinkerOptions& options,
                            const std::vector<Module*>& modules) {
  assert(_variants.empty());
  assert(modules.size() == options.GetInFiles().size());

  for (size_t i = 0; i < modules.size(); ++i) {
    const auto* feat_mgr = modules[i]->context()->get_feature_mgr();
    if ((feat_mgr->HasCapability(spv::Capability::FunctionVariantsINTEL)) ||
        (feat_mgr->HasCapability(spv::Capability::SpecConditionalINTEL)) ||
        (feat_mgr->HasExtension(kSPV_INTEL_function_variants))) {
      // In principle, it can be done but it's complicated due to having to
      // combine the existing conditionals with the new ones. For example,
      // conditional capabilities would need to become "doubly-conditional".
      _err << "Creating multitarget modules from multitarget modules is not "
              "supported. Offending file: "
           << options.GetInFiles()[i];
      return false;
    }
  }

  std::vector<std::vector<std::string>> target_rows;
  std::vector<std::vector<std::string>> architecture_rows;

  if (!options.GetFnVarTargetsCsv().empty()) {
    const std::vector<std::string> tgt_cols = {"module", "target", "features"};
    if (!ParseCsv(options.GetFnVarTargetsCsv(), tgt_cols, _err, target_rows)) {
      return false;
    }
  }

  if (!options.GetFnVarArchitecturesCsv().empty()) {
    const std::vector<std::string> arch_cols = {"module", "category", "family",
                                                "op", "architecture"};
    if (!ParseCsv(options.GetFnVarArchitecturesCsv(), arch_cols, _err,
                  architecture_rows)) {
      return false;
    }
  }

  // check that all modules defined in the CSV exist

  for (const auto& tgt_vals : target_rows) {
    bool found = false;
    for (const auto& in_file : options.GetInFiles()) {
      if (tgt_vals[0] == in_file) {
        found = true;
      }
    }
    if (!found) {
      _err << "Module '" << tgt_vals[0]
           << "' found in targets CSV not passed to the CLI.";
      return false;
    }
  }

  for (const auto& arch_vals : architecture_rows) {
    bool found = false;
    for (const auto& in_file : options.GetInFiles()) {
      if (arch_vals[0] == in_file) {
        found = true;
      }
    }
    if (!found) {
      _err << "Module '" << arch_vals[0]
           << "' found in architectures CSV not passed to the CLI.";
      return false;
    }
  }

  // create per-module variants

  for (size_t i = 0; i < modules.size(); ++i) {
    // first module passed to the CLI is considered the base module
    bool is_base = i == 0;
    const auto name = options.GetInFiles()[i];
    auto variant = Variant(is_base, name, modules[i]);

    for (const auto& arch_row : architecture_rows) {
      const auto row_name = arch_row[0];
      if (row_name == name) {
        uint32_t category, family, op, architecture;

        if (!(std::stringstream(arch_row[1]) >> category)) {
          _err << "Error converting " << arch_row[1]
               << " to architecture category.";
          return false;
        }
        if (!(std::stringstream(arch_row[2]) >> family)) {
          _err << "Error converting " << arch_row[2]
               << " to architecture family.";
          return false;
        }
        if (!(std::stringstream(arch_row[3]) >> op)) {
          _err << "Error converting " << arch_row[3] << " to architecture op.";
          return false;
        }
        if (!(std::stringstream(arch_row[4]) >> architecture)) {
          _err << "Error converting " << arch_row[4] << " to architecture.";
          return false;
        }

        variant.AddArchDef(category, family, op, architecture);
      }
    }

    for (const auto& tgt_row : target_rows) {
      const auto row_name = tgt_row[0];
      if (row_name == name) {
        uint32_t target;
        std::vector<uint32_t> features;

        if (!(std::stringstream(tgt_row[1]) >> target)) {
          _err << "Error converting " << tgt_row[1] << " to target.";
          return false;
        }

        // get features as FEAT_SEP-delimited integers

        std::stringstream feat_stream(tgt_row[2]);
        std::string feat;
        while (std::getline(feat_stream, feat, FEAT_SEP)) {
          uint32_t ufeat;
          if (!(std::stringstream(feat) >> ufeat)) {
            _err << "Error converting " << tgt_row[2] << " to target features.";
            return false;
          }
          features.push_back(ufeat);
        }

        variant.AddTgtDef(target, features);
      }
    }

    if (options.GetFnVarCapabilities()) {
      variant.InferCapabilities();
    }

    _variants.push_back(variant);
  }

  return true;
}

bool Variants::ProcessVariants() {
  EnsureBoolType();
  CollectVarInsts();
  if (!GenerateFnVarConstants()) {
    return false;
  }
  CollectBaseFnCalls();
  return true;
}

void Variants::GenerateHeader(IRContext* linked_context) {
  linked_context->AddCapability(spv::Capability::SpecConditionalINTEL);
  linked_context->AddCapability(spv::Capability::FunctionVariantsINTEL);
  linked_context->AddExtension(FNVAR_EXT_NAME);

  // Specifies used registry version
  auto inst = Instruction(linked_context, spv::Op::OpModuleProcessed);
  std::stringstream line;
  line << "SPV_INTEL_function_variants registry version "
       << FNVAR_REGISTRY_VERSION;
  inst.AddOperand(
      {SPV_OPERAND_TYPE_LITERAL_STRING, utils::MakeVector(line.str())});
  linked_context->AddDebug3Inst(
      std::unique_ptr<Instruction>(inst.Clone(linked_context)));
}

void Variants::CombineVariantInstructions(IRContext* linked_context) {
  CombineBaseFnCalls(linked_context);
  CombineInstructions(linked_context);
}

void Variants::EnsureBoolType() {
  for (auto& variant : _variants) {
    Module* module = variant.GetModule();
    IRContext* context = module->context();

    uint32_t bool_id = FindIdOfBoolType(module);
    if (bool_id == 0) {
      bool_id = context->TakeNextId();
      auto variant_bool =
          Instruction(context, spv::Op::OpTypeBool, 0, bool_id, {});
      module->AddType(
          std::unique_ptr<Instruction>(variant_bool.Clone(context)));
    }
  }
}

void Variants::CollectVarInsts() {
  for (size_t i = 0; i < _variants.size(); ++i) {
    const auto variant = _variants[i];
    const auto* var_mod = variant.GetModule();

    var_mod->ForEachInst([this, &i](const Instruction* inst) {
      if (CanBeFnVarCombined(inst)) {
        const size_t inst_hash = HashInst(inst);
        if (_fnvar_usage.find(inst_hash) == _fnvar_usage.end()) {
          _fnvar_usage.insert({inst_hash, {i}});
        } else {
          assert(_fnvar_usage[inst_hash].size() < _variants.size());
          _fnvar_usage[inst_hash].push_back(i);
        }
      }
    });
  }
}

bool Variants::GenerateFnVarConstants() {
  assert(_variants.size() > 0);
  assert(_variants[0].IsBase());

  if (_variants.size() == 1) {
    return true;
  }

  for (auto& variant : _variants) {
    Module* module = variant.GetModule();
    IRContext* context = module->context();

    uint32_t bool_id = FindIdOfBoolType(module);
    if (bool_id == 0) {
      // add a bool type if not present already
      bool_id = context->TakeNextId();
      auto variant_bool =
          Instruction(context, spv::Op::OpTypeBool, 0, bool_id, {});
      module->AddType(
          std::unique_ptr<Instruction>(variant_bool.Clone(context)));
    }

    // Spec constant architecture and target

    std::vector<uint32_t> spec_const_arch_ids;
    for (const auto& arch_def : variant.GetArchDefs()) {
      const uint32_t spec_const_arch_id = context->TakeNextId();
      spec_const_arch_ids.push_back(spec_const_arch_id);

      const auto inst = Instruction(
          context, spv::Op::OpSpecConstantArchitectureINTEL, bool_id,
          spec_const_arch_id,
          std::initializer_list<opt::Operand>{
              {SPV_OPERAND_TYPE_LITERAL_INTEGER, {arch_def.category}},
              {SPV_OPERAND_TYPE_LITERAL_INTEGER, {arch_def.family}},
              // Using spec op opcode here expects then next operand to be
              // a type:
              {SPV_OPERAND_TYPE_LITERAL_INTEGER, {arch_def.op}},
              {SPV_OPERAND_TYPE_LITERAL_INTEGER, {arch_def.architecture}},
          });
      module->AddType(std::unique_ptr<Instruction>(inst.Clone(context)));
    }

    std::vector<uint32_t> spec_const_tgt_ids;
    for (const auto& tgt_def : variant.GetTgtDefs()) {
      const uint32_t spec_const_tgt_id = context->TakeNextId();
      spec_const_tgt_ids.push_back(spec_const_tgt_id);

      auto inst =
          Instruction(context, spv::Op::OpSpecConstantTargetINTEL, bool_id,
                      spec_const_tgt_id,
                      std::initializer_list<opt::Operand>{
                          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {tgt_def.target}},
                      });
      for (const auto& feat : tgt_def.features) {
        inst.AddOperand({SPV_OPERAND_TYPE_LITERAL_INTEGER, {feat}});
      }
      module->AddType(std::unique_ptr<Instruction>(inst.Clone(context)));
    }

    std::vector<uint32_t> spec_const_ids;

    // Spec constant capabilities

    const auto variant_capabilities = variant.GetCapabilities();
    if (!variant_capabilities.empty()) {
      const uint32_t spec_const_cap_id = context->TakeNextId();
      auto inst = Instruction(context, spv::Op::OpSpecConstantCapabilitiesINTEL,
                              bool_id, spec_const_cap_id, {});
      for (const auto& cap : variant_capabilities) {
        inst.AddOperand({SPV_OPERAND_TYPE_CAPABILITY, {uint32_t(cap)}});
      }
      module->AddType(std::unique_ptr<Instruction>(inst.Clone(context)));
      spec_const_ids.push_back(spec_const_cap_id);
    }

    // Combine architectures such that, for the same module, those with the same
    // category and family are combined with AND and different cat/fam are
    // combined with OR.
    // This lets you create combinations like "architecture between X and Y".

    // map (category, family) -> IDs
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> arch_map_and;

    for (size_t i = 0; i < spec_const_arch_ids.size(); ++i) {
      const auto& arch_def = variant.GetArchDefs()[i];
      const auto id = spec_const_arch_ids[i];
      const auto key = std::make_pair(arch_def.category, arch_def.family);
      if (arch_map_and.find(key) == arch_map_and.end()) {
        arch_map_and[key] = {id};
      } else {
        arch_map_and[key].push_back(id);
      }
    }

    std::vector<uint32_t> arch_ids_or;
    for (const auto& it : arch_map_and) {
      const auto id = CombineIds(context, it.second, spv::Op::OpLogicalAnd);
      if (id > 0) {
        arch_ids_or.push_back(id);
      }
    }

    const uint32_t spec_const_arch_id =
        CombineIds(context, arch_ids_or, spv::Op::OpLogicalOr);
    if (spec_const_arch_id > 0) {
      spec_const_ids.push_back(spec_const_arch_id);
    }

    const uint32_t spec_const_tgt_id =
        CombineIds(context, spec_const_tgt_ids, spv::Op::OpLogicalOr);
    if (spec_const_tgt_id > 0) {
      spec_const_ids.push_back(spec_const_tgt_id);
    }

    uint32_t combined_spec_const_id =
        CombineIds(context, spec_const_ids, spv::Op::OpLogicalAnd);
    if (combined_spec_const_id == 0) {
      // If the variant module has no constraints, use SpecConstantTrue
      combined_spec_const_id = context->TakeNextId();
      const auto inst = Instruction(context, spv::Op::OpSpecConstantTrue,
                                    bool_id, combined_spec_const_id,
                                    std::initializer_list<opt::Operand>{});
      context->module()->AddType(
          std::unique_ptr<Instruction>(inst.Clone(context)));
    }
    assert(combined_spec_const_id != 0);

    // Add a name the combined boolean ID so we can look it up after the IDs are
    // shifted
    auto inst = Instruction(context, spv::Op::OpName);
    inst.AddOperand({SPV_OPERAND_TYPE_ID, {combined_spec_const_id}});
    std::vector<uint32_t> str_words;
    utils::AppendToVector(variant.GetName(), &str_words);
    inst.AddOperand({SPV_OPERAND_TYPE_LITERAL_STRING, {str_words}});
    module->AddDebug2Inst(std::unique_ptr<Instruction>(inst.Clone(context)));

    // Annotate all instructions in the types section (eg. constants) with
    // ConditionalINTEL, unless they can be shared between _variants (eg.
    // types). Spec constants are excluded because they might have been
    // generated by this extension.
    for (const auto& type_inst : module->types_values()) {
      if (!CanBeFnVarCombined(&type_inst) &&
          !spvOpcodeIsSpecConstant(type_inst.opcode())) {
        DecorateConditional(context, type_inst.result_id(),
                            combined_spec_const_id);
      }
    }
  }

  // Annotate functions with ConditionalINTEL

  for (const auto& base_fn : *_variants[0].GetModule()) {
    // For each function of the base module, find matching variant functions in
    // other modules

    auto base_fn_name = GetFnName(base_fn.DefInst());
    if (base_fn_name.empty()) {
      _err << "Could not find name of a function " << base_fn.result_id()
           << " in a base module " << _variants[0].GetName()
           << ". To be usable by SPV_INTEL_function__variants, a function "
              "must either have an entry point or an export "
              "LinkAttribute decoration.";
      return false;
    }

    bool base_fn_needs_conditional = false;
    for (size_t i = 1; i < _variants.size(); ++i) {
      const auto& variant = _variants[i];
      auto* variant_module = variant.GetModule();
      auto* variant_context = variant_module->context();

      for (const auto& var_fn : *variant_module) {
        auto var_fn_name = GetFnName(var_fn.DefInst());
        if (var_fn_name.empty()) {
          _err << "Could not find name of a function " << var_fn.result_id()
               << " in a base module " << variant.GetName()
               << ". To be usable by SPV_INTEL_function__variants, a function "
                  "must either have an entry point or an export "
                  "LinkAttribute decoration.";
          return false;
        }

        if (base_fn_name == var_fn_name) {
          base_fn_needs_conditional = true;
        }

        // each function in a variant module gets a ConditionalINTEL decoration

        uint32_t spec_const_id =
            FindSpecConstByName(variant_module, variant.GetName());
        assert(spec_const_id != 0);
        DecorateConditional(variant_context, var_fn.result_id(), spec_const_id);
        ConvertEPToConditional(variant_module, var_fn, spec_const_id);
      }
    }

    if (base_fn_needs_conditional) {
      // only a base function that has a variant in another module gets a
      // ConditionalINTEL decoration, the others are common for all _variants
      auto* base_module = _variants[0].GetModule();
      auto* base_context = base_module->context();
      uint32_t spec_const_id =
          FindSpecConstByName(base_module, _variants[0].GetName());
      assert(spec_const_id != 0);
      DecorateConditional(base_context, base_fn.result_id(), spec_const_id);
      ConvertEPToConditional(base_module, base_fn, spec_const_id);
    }
  }

  return true;
}

void Variants::CollectBaseFnCalls() {
  auto* base_mod = _variants[0].GetModule();
  assert(_variants[0].IsBase());
  const auto* base_def_use_mgr = base_mod->context()->get_def_use_mgr();

  base_mod->ForEachInst([this, &base_def_use_mgr](const Instruction* inst) {
    if (inst->opcode() == spv::Op::OpFunctionCall) {
      // For each function call in base module, get the function name
      const auto fn_id = inst->GetOperand(2).AsId();
      const auto* called_fn_inst = base_def_use_mgr->GetDef(fn_id);
      assert(called_fn_inst != nullptr);
      const auto called_fn_name = GetFnName(*called_fn_inst);
      assert(!called_fn_name.empty());

      std::vector<std::pair<std::string, const opt::Function*>> called_fns;
      for (size_t i = 1; i < _variants.size(); ++i) {
        // ... then see in which variant the called function was defined
        const auto& variant = _variants[i];
        assert(!variant.IsBase());

        for (const auto& fn : *variant.GetModule()) {
          const auto fn_name = GetFnName(fn.DefInst());
          if (fn_name == called_fn_name) {
            called_fns.push_back(std::make_pair(variant.GetName(), &fn));
          }
        }
      }

      if (!called_fns.empty()) {
        _base_fn_calls[inst->result_id()] = called_fns;
      }
    }
  });
}

void Variants::CombineBaseFnCalls(IRContext* linked_context) {
  for (auto kv : _base_fn_calls) {
    const uint32_t call_id = kv.first;
    const auto called_fns = kv.second;

    if (called_fns.empty()) {
      return;
    }

    opt::BasicBlock* fn_call_bb = linked_context->get_instr_block(call_id);

    Instruction* found_call_inst = nullptr;
    auto bb_iter = fn_call_bb->begin();
    while (bb_iter != fn_call_bb->end() && found_call_inst == nullptr) {
      if (bb_iter->HasResultId() && bb_iter->result_id() == call_id) {
        found_call_inst = &*bb_iter;
      }
      ++bb_iter;
    }

    if (found_call_inst == nullptr) {
      return;
    }

    const auto base_spec_const_id =
        FindSpecConstByName(_variants[0].GetModule(), _variants[0].GetName());
    const auto base_type_op = found_call_inst->context()
                                  ->get_def_use_mgr()
                                  ->GetDef(found_call_inst->type_id())
                                  ->opcode();
    const auto base_call_id = found_call_inst->result_id();

    // decorate the base call with ConditionalINTEL
    DecorateConditional(linked_context, base_call_id, base_spec_const_id);

    // Add OpFunctionCall for each variant
    Instruction* last_inst = found_call_inst;
    std::vector<std::pair<uint32_t, uint32_t>> var_call_ids;
    for (const auto& kv2 : called_fns) {
      const std::string var_name = kv2.first;
      const opt::Function* fn = kv2.second;
      const uint32_t spec_const_id =
          FindSpecConstByName(linked_context->module(), var_name);
      assert(spec_const_id != 0);
      const uint32_t var_call_id = linked_context->TakeNextId();
      var_call_ids.push_back(std::make_pair(spec_const_id, var_call_id));

      auto* var_call_inst = found_call_inst->Clone(linked_context);
      var_call_inst->SetResultId(var_call_id);
      var_call_inst->SetOperand(2, {fn->result_id()});
      var_call_inst->InsertAfter(last_inst);
      linked_context->set_instr_block(var_call_inst, fn_call_bb);
      last_inst = var_call_inst;

      // decorate the variant call with ConditionalINTEL
      DecorateConditional(linked_context, var_call_id, spec_const_id);
    }

    if (base_type_op != spv::Op::OpTypeVoid) {
      // Add OpConditionalCopyObjectINTEL combining the function calls
      const uint32_t result_id = linked_context->TakeNextId();
      auto conditional_copy_inst = new Instruction(
          linked_context, spv::Op::OpConditionalCopyObjectINTEL,
          found_call_inst->type_id(), result_id,
          {{SPV_OPERAND_TYPE_ID, {base_spec_const_id}},
           {SPV_OPERAND_TYPE_ID, {found_call_inst->result_id()}}});

      for (const auto& kv3 : var_call_ids) {
        const auto spec_const_id = kv3.first;
        const auto var_call_id = kv3.second;
        conditional_copy_inst->AddOperand(
            {SPV_OPERAND_TYPE_ID, {spec_const_id}});
        conditional_copy_inst->AddOperand({SPV_OPERAND_TYPE_ID, {var_call_id}});
      }
      conditional_copy_inst->InsertAfter(last_inst);
      linked_context->set_instr_block(conditional_copy_inst, fn_call_bb);
      last_inst = conditional_copy_inst;

      // In all remaining instructions within the basic block, replace all
      // usages of the base call ID with the result of
      // OpConditionalCopyObjectINTEL
      do {
        last_inst = last_inst->NextNode();
        last_inst->ForEachInId([base_call_id, result_id](uint32_t* id) {
          if (*id == base_call_id) {
            *id = result_id;
          }
        });
      } while (last_inst != nullptr && *last_inst != *fn_call_bb->tail());
    }
  }

  // Combine spec consts for the base module (base module is activated if all
  // variants are false AND the base module constraints are satisfied)

  std::vector<uint32_t> var_spec_const_ids;
  for (const auto& variant : _variants) {
    if (variant.IsBase()) {
      continue;
    }

    const auto id =
        FindSpecConstByName(linked_context->module(), variant.GetName());
    assert(id != 0);
    var_spec_const_ids.push_back(id);
  }
  const uint32_t base_or_id =
      CombineIds(linked_context, var_spec_const_ids, spv::Op::OpLogicalOr);

  if (base_or_id != 0) {
    const uint32_t bool_id = FindIdOfBoolType(linked_context->module());
    assert(bool_id != 0);

    const uint32_t base_not_id = linked_context->TakeNextId();
    const auto spec_const_op_inst = Instruction(
        linked_context, spv::Op::OpSpecConstantOp, bool_id, base_not_id,
        std::initializer_list<opt::Operand>{
            {SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER,
             {(uint32_t)(spv::Op::OpLogicalNot)}},
            {SPV_OPERAND_TYPE_ID, {base_or_id}}});
    linked_context->module()->AddType(
        std::unique_ptr<Instruction>(spec_const_op_inst.Clone(linked_context)));

    // Update any ConditionalINTEL annotations, names and entry points
    // referencing the old spec const ID to use the new one

    const uint32_t old_base_spec_const_id =
        FindSpecConstByName(linked_context->module(), _variants[0].GetName());
    assert(old_base_spec_const_id != 0);
    const uint32_t base_spec_const_id =
        CombineIds(linked_context, {old_base_spec_const_id, base_not_id},
                   spv::Op::OpLogicalAnd);

    for (auto& annot_inst : linked_context->module()->annotations()) {
      if ((annot_inst.GetOperand(1).words[0] ==
           uint32_t(spv::Decoration::ConditionalINTEL)) &&
          (annot_inst.GetOperand(2).AsId() == old_base_spec_const_id)) {
        annot_inst.SetOperand(2, {base_spec_const_id});
      }
    }

    for (auto& name_inst : linked_context->module()->debugs2()) {
      if ((name_inst.opcode() == spv::Op::OpName) &&
          (name_inst.GetOperand(0).AsId() == old_base_spec_const_id)) {
        name_inst.SetOperand(0, {base_spec_const_id});
      }
    }

    for (auto& ep_inst : linked_context->module()->entry_points()) {
      if ((ep_inst.opcode() == spv::Op::OpConditionalEntryPointINTEL) &&
          (ep_inst.GetOperand(0).AsId() == old_base_spec_const_id)) {
        ep_inst.SetOperand(0, {base_spec_const_id});
      }
    }

    linked_context->module()->ForEachInst(
        [old_base_spec_const_id, base_spec_const_id](Instruction* inst) {
          if (inst->opcode() == spv::Op::OpConditionalCopyObjectINTEL) {
            inst->ForEachInId(
                [old_base_spec_const_id, base_spec_const_id](uint32_t* id) {
                  if (*id == old_base_spec_const_id) {
                    *id = base_spec_const_id;
                  }
                });
          }
        });
  }
}

void Variants::CombineInstructions(IRContext* linked_context) {
  // cache for existing variant ID combinations
  std::map<std::vector<size_t>, uint32_t> spec_const_comb_ids;

  linked_context->module()->ForEachInst(
      [this, &linked_context, &spec_const_comb_ids](Instruction* inst) {
        if (!CanBeFnVarCombined(inst)) {
          return;
        }

        const size_t inst_hash = HashInst(inst);
        if (_fnvar_usage.find(inst_hash) != _fnvar_usage.end()) {
          const std::vector<size_t> var_ids = _fnvar_usage[inst_hash];
          const uint32_t spec_const_comb_id = CombineVariants(
              _variants, var_ids, linked_context, spec_const_comb_ids);
          if (spec_const_comb_id != 0) {
            if (inst->HasResultId()) {
              DecorateConditional(linked_context, inst->result_id(),
                                  spec_const_comb_id);
            } else if (inst->opcode() == spv::Op::OpCapability) {
              const uint32_t cap = inst->GetOperand(0).words[0];
              inst->SetOpcode(spv::Op::OpConditionalCapabilityINTEL);
              inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {spec_const_comb_id}},
                                   {SPV_OPERAND_TYPE_CAPABILITY, {cap}}});
            } else if (inst->opcode() == spv::Op::OpExtension) {
              const std::string ext_name = inst->GetOperand(0).AsString();
              inst->SetOpcode(spv::Op::OpConditionalExtensionINTEL);
              inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {spec_const_comb_id}},
                                   {SPV_OPERAND_TYPE_LITERAL_STRING,
                                    {utils::MakeVector(ext_name)}}});
            } else {
              assert(false && "Unsupported");
            }
          }
        }
      });
}

}  // namespace spvtools
