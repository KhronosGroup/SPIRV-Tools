// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/fact_manager.h"

#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "source/fuzz/equivalence_relation.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/uniform_buffer_element_descriptor.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

namespace {

std::string ToString(const protobufs::Fact& fact) {
  assert(fact.fact_case() == protobufs::Fact::kConstantUniformFact &&
         "Right now this is the only fact we know how to stringify.");
  std::stringstream stream;
  stream << "("
         << fact.constant_uniform_fact()
                .uniform_buffer_element_descriptor()
                .descriptor_set()
         << ", "
         << fact.constant_uniform_fact()
                .uniform_buffer_element_descriptor()
                .binding()
         << ")[";

  bool first = true;
  for (auto index : fact.constant_uniform_fact()
                        .uniform_buffer_element_descriptor()
                        .index()) {
    if (first) {
      first = false;
    } else {
      stream << ", ";
    }
    stream << index;
  }

  stream << "] == [";

  first = true;
  for (auto constant_word : fact.constant_uniform_fact().constant_word()) {
    if (first) {
      first = false;
    } else {
      stream << ", ";
    }
    stream << constant_word;
  }

  stream << "]";
  return stream.str();
}

}  // namespace

//=======================
// Constant uniform facts

// The purpose of this class is to group the fields and data used to represent
// facts about uniform constants.
class FactManager::ConstantUniformFacts {
 public:
  // See method in FactManager which delegates to this method.
  bool AddFact(const protobufs::FactConstantUniform& fact,
               opt::IRContext* context);

  // See method in FactManager which delegates to this method.
  std::vector<uint32_t> GetConstantsAvailableFromUniformsForType(
      opt::IRContext* ir_context, uint32_t type_id) const;

  // See method in FactManager which delegates to this method.
  const std::vector<protobufs::UniformBufferElementDescriptor>
  GetUniformDescriptorsForConstant(opt::IRContext* ir_context,
                                   uint32_t constant_id) const;

  // See method in FactManager which delegates to this method.
  uint32_t GetConstantFromUniformDescriptor(
      opt::IRContext* context,
      const protobufs::UniformBufferElementDescriptor& uniform_descriptor)
      const;

  // See method in FactManager which delegates to this method.
  std::vector<uint32_t> GetTypesForWhichUniformValuesAreKnown() const;

  // See method in FactManager which delegates to this method.
  const std::vector<std::pair<protobufs::FactConstantUniform, uint32_t>>&
  GetConstantUniformFactsAndTypes() const;

 private:
  // Returns true if and only if the words associated with
  // |constant_instruction| exactly match the words for the constant associated
  // with |constant_uniform_fact|.
  bool DataMatches(
      const opt::Instruction& constant_instruction,
      const protobufs::FactConstantUniform& constant_uniform_fact) const;

  // Yields the constant words associated with |constant_uniform_fact|.
  std::vector<uint32_t> GetConstantWords(
      const protobufs::FactConstantUniform& constant_uniform_fact) const;

  // Yields the id of a constant of type |type_id| whose data matches the
  // constant data in |constant_uniform_fact|, or 0 if no such constant is
  // declared.
  uint32_t GetConstantId(
      opt::IRContext* context,
      const protobufs::FactConstantUniform& constant_uniform_fact,
      uint32_t type_id) const;

  // Checks that the width of a floating-point constant is supported, and that
  // the constant is finite.
  bool FloatingPointValueIsSuitable(const protobufs::FactConstantUniform& fact,
                                    uint32_t width) const;

  std::vector<std::pair<protobufs::FactConstantUniform, uint32_t>>
      facts_and_type_ids_;
};

uint32_t FactManager::ConstantUniformFacts::GetConstantId(
    opt::IRContext* context,
    const protobufs::FactConstantUniform& constant_uniform_fact,
    uint32_t type_id) const {
  auto type = context->get_type_mgr()->GetType(type_id);
  assert(type != nullptr && "Unknown type id.");
  auto constant = context->get_constant_mgr()->GetConstant(
      type, GetConstantWords(constant_uniform_fact));
  return context->get_constant_mgr()->FindDeclaredConstant(constant, type_id);
}

std::vector<uint32_t> FactManager::ConstantUniformFacts::GetConstantWords(
    const protobufs::FactConstantUniform& constant_uniform_fact) const {
  std::vector<uint32_t> result;
  for (auto constant_word : constant_uniform_fact.constant_word()) {
    result.push_back(constant_word);
  }
  return result;
}

bool FactManager::ConstantUniformFacts::DataMatches(
    const opt::Instruction& constant_instruction,
    const protobufs::FactConstantUniform& constant_uniform_fact) const {
  assert(constant_instruction.opcode() == SpvOpConstant);
  std::vector<uint32_t> data_in_constant;
  for (uint32_t i = 0; i < constant_instruction.NumInOperands(); i++) {
    data_in_constant.push_back(constant_instruction.GetSingleWordInOperand(i));
  }
  return data_in_constant == GetConstantWords(constant_uniform_fact);
}

std::vector<uint32_t>
FactManager::ConstantUniformFacts::GetConstantsAvailableFromUniformsForType(
    opt::IRContext* ir_context, uint32_t type_id) const {
  std::vector<uint32_t> result;
  std::set<uint32_t> already_seen;
  for (auto& fact_and_type_id : facts_and_type_ids_) {
    if (fact_and_type_id.second != type_id) {
      continue;
    }
    if (auto constant_id =
            GetConstantId(ir_context, fact_and_type_id.first, type_id)) {
      if (already_seen.find(constant_id) == already_seen.end()) {
        result.push_back(constant_id);
        already_seen.insert(constant_id);
      }
    }
  }
  return result;
}

const std::vector<protobufs::UniformBufferElementDescriptor>
FactManager::ConstantUniformFacts::GetUniformDescriptorsForConstant(
    opt::IRContext* ir_context, uint32_t constant_id) const {
  std::vector<protobufs::UniformBufferElementDescriptor> result;
  auto constant_inst = ir_context->get_def_use_mgr()->GetDef(constant_id);
  assert(constant_inst->opcode() == SpvOpConstant &&
         "The given id must be that of a constant");
  auto type_id = constant_inst->type_id();
  for (auto& fact_and_type_id : facts_and_type_ids_) {
    if (fact_and_type_id.second != type_id) {
      continue;
    }
    if (DataMatches(*constant_inst, fact_and_type_id.first)) {
      result.emplace_back(
          fact_and_type_id.first.uniform_buffer_element_descriptor());
    }
  }
  return result;
}

uint32_t FactManager::ConstantUniformFacts::GetConstantFromUniformDescriptor(
    opt::IRContext* context,
    const protobufs::UniformBufferElementDescriptor& uniform_descriptor) const {
  // Consider each fact.
  for (auto& fact_and_type : facts_and_type_ids_) {
    // Check whether the uniform descriptor associated with the fact matches
    // |uniform_descriptor|.
    if (UniformBufferElementDescriptorEquals()(
            &uniform_descriptor,
            &fact_and_type.first.uniform_buffer_element_descriptor())) {
      return GetConstantId(context, fact_and_type.first, fact_and_type.second);
    }
  }
  // No fact associated with the given uniform descriptor was found.
  return 0;
}

std::vector<uint32_t>
FactManager::ConstantUniformFacts::GetTypesForWhichUniformValuesAreKnown()
    const {
  std::vector<uint32_t> result;
  for (auto& fact_and_type : facts_and_type_ids_) {
    if (std::find(result.begin(), result.end(), fact_and_type.second) ==
        result.end()) {
      result.push_back(fact_and_type.second);
    }
  }
  return result;
}

bool FactManager::ConstantUniformFacts::FloatingPointValueIsSuitable(
    const protobufs::FactConstantUniform& fact, uint32_t width) const {
  const uint32_t kFloatWidth = 32;
  const uint32_t kDoubleWidth = 64;
  if (width != kFloatWidth && width != kDoubleWidth) {
    // Only 32- and 64-bit floating-point types are handled.
    return false;
  }
  std::vector<uint32_t> words = GetConstantWords(fact);
  if (width == 32) {
    float value;
    memcpy(&value, words.data(), sizeof(float));
    if (!std::isfinite(value)) {
      return false;
    }
  } else {
    double value;
    memcpy(&value, words.data(), sizeof(double));
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

bool FactManager::ConstantUniformFacts::AddFact(
    const protobufs::FactConstantUniform& fact, opt::IRContext* context) {
  // Try to find a unique instruction that declares a variable such that the
  // variable is decorated with the descriptor set and binding associated with
  // the constant uniform fact.
  opt::Instruction* uniform_variable = FindUniformVariable(
      fact.uniform_buffer_element_descriptor(), context, true);

  if (!uniform_variable) {
    return false;
  }

  assert(SpvOpVariable == uniform_variable->opcode());
  assert(SpvStorageClassUniform == uniform_variable->GetSingleWordInOperand(0));

  auto should_be_uniform_pointer_type =
      context->get_type_mgr()->GetType(uniform_variable->type_id());
  if (!should_be_uniform_pointer_type->AsPointer()) {
    return false;
  }
  if (should_be_uniform_pointer_type->AsPointer()->storage_class() !=
      SpvStorageClassUniform) {
    return false;
  }
  auto should_be_uniform_pointer_instruction =
      context->get_def_use_mgr()->GetDef(uniform_variable->type_id());
  auto composite_type =
      should_be_uniform_pointer_instruction->GetSingleWordInOperand(1);

  auto final_element_type_id = fuzzerutil::WalkCompositeTypeIndices(
      context, composite_type,
      fact.uniform_buffer_element_descriptor().index());
  if (!final_element_type_id) {
    return false;
  }
  auto final_element_type =
      context->get_type_mgr()->GetType(final_element_type_id);
  assert(final_element_type &&
         "There should be a type corresponding to this id.");

  if (!(final_element_type->AsFloat() || final_element_type->AsInteger())) {
    return false;
  }
  auto width = final_element_type->AsFloat()
                   ? final_element_type->AsFloat()->width()
                   : final_element_type->AsInteger()->width();

  if (final_element_type->AsFloat() &&
      !FloatingPointValueIsSuitable(fact, width)) {
    return false;
  }

  auto required_words = (width + 32 - 1) / 32;
  if (static_cast<uint32_t>(fact.constant_word().size()) != required_words) {
    return false;
  }
  facts_and_type_ids_.emplace_back(
      std::pair<protobufs::FactConstantUniform, uint32_t>(
          fact, final_element_type_id));
  return true;
}

const std::vector<std::pair<protobufs::FactConstantUniform, uint32_t>>&
FactManager::ConstantUniformFacts::GetConstantUniformFactsAndTypes() const {
  return facts_and_type_ids_;
}

// End of uniform constant facts
//==============================

//==============================
// Data synonym facts

// The purpose of this class is to group the fields and data used to represent
// facts about data synonyms.
class FactManager::DataSynonymFacts {
 public:
  // See method in FactManager which delegates to this method.
  void AddFact(const protobufs::FactDataSynonym& fact, opt::IRContext* context);

  // See method in FactManager which delegates to this method.
  std::vector<const protobufs::DataDescriptor*> GetSynonymsForDataDescriptor(
      const protobufs::DataDescriptor& data_descriptor,
      opt::IRContext* context) const;

  // See method in FactManager which delegates to this method.
  std::vector<uint32_t> GetIdsForWhichSynonymsAreKnown(
      opt::IRContext* context) const;

  // See method in FactManager which delegates to this method.
  bool IsSynonymous(const protobufs::DataDescriptor& data_descriptor1,
                    const protobufs::DataDescriptor& data_descriptor2,
                    opt::IRContext* context) const;

 private:
  // Adds |fact| to the set of managed facts, and recurses into sub-components
  // of the data descriptors referenced in |fact|, if they are composites, to
  // record that their components are pairwise-synonymous.
  void AddFactRecursive(const protobufs::FactDataSynonym& fact,
                        opt::IRContext* context);

  // Inspects all known facts and adds corollary facts; e.g. if we know that
  // a.x == b.x and a.y == b.y, where a and b have vec2 type, we can record
  // that a == b holds.
  //
  // This method is expensive, and is thus called on demand: rather than
  // computing the closure of facts each time a data synonym fact is added, we
  // compute the closure only when a data synonym fact is *queried*.
  void ComputeClosureOfFacts(opt::IRContext* context);

  EquivalenceRelation<protobufs::DataDescriptor, DataDescriptorHash,
                      DataDescriptorEquals>
      synonymous_;
  bool closure_computation_required = false;
};

void FactManager::DataSynonymFacts::AddFact(
    const protobufs::FactDataSynonym& fact, opt::IRContext* context) {
  // Add the fact, including all facts relating sub-components of the data
  // descriptors that are involved.
  AddFactRecursive(fact, context);
}

void FactManager::DataSynonymFacts::AddFactRecursive(
    const protobufs::FactDataSynonym& fact, opt::IRContext* context) {
  closure_computation_required = true;
  synonymous_.MakeEquivalent(fact.data1(), fact.data2());

  uint32_t type_id = fuzzerutil::WalkCompositeTypeIndices(
      context,
      context->get_def_use_mgr()->GetDef(fact.data1().object())->type_id(),
      fact.data1().index());

  auto type = context->get_type_mgr()->GetType(type_id);
  auto type_instruction = context->get_def_use_mgr()->GetDef(type_id);
  assert(type != nullptr &&
         "Invalid data synonym fact: one side has an "
         "unknown type.");
  uint32_t num_composite_elements;
  if (type->AsArray()) {
    num_composite_elements =
        fuzzerutil::GetArraySize(*type_instruction, context);
  } else if (type->AsMatrix()) {
    num_composite_elements = type->AsMatrix()->element_count();
  } else if (type->AsStruct()) {
    num_composite_elements =
        fuzzerutil::GetNumberOfStructMembers(*type_instruction);
  } else if (type->AsVector()) {
    num_composite_elements = type->AsVector()->element_count();
  } else {
    return;
  }
  for (uint32_t i = 0; i < num_composite_elements; i++) {
    std::vector<uint32_t> extended_indices1 =
        fuzzerutil::RepeatedFieldToVector(fact.data1().index());
    extended_indices1.push_back(i);
    std::vector<uint32_t> extended_indices2 =
        fuzzerutil::RepeatedFieldToVector(fact.data1().index());
    extended_indices2.push_back(i);
    protobufs::FactDataSynonym extended_data_synonym_fact;
    *extended_data_synonym_fact.mutable_data1() =
        MakeDataDescriptor(fact.data1().object(), std::move(extended_indices1));
    *extended_data_synonym_fact.mutable_data2() =
        MakeDataDescriptor(fact.data2().object(), std::move(extended_indices2));
    AddFact(extended_data_synonym_fact, context);
  }
}

void FactManager::DataSynonymFacts::ComputeClosureOfFacts(
    opt::IRContext* context) {
  using DataDescriptorPair =
      std::pair<protobufs::DataDescriptor, protobufs::DataDescriptor>;

  struct DataDescriptorPairHash {
    std::size_t operator()(const DataDescriptorPair& pair) const {
      return DataDescriptorHash()(&pair.first) ^
             DataDescriptorHash()(&pair.second);
    }
  };

  struct DataDescriptorPairEquals {
    bool operator()(const DataDescriptorPair& first,
                    const DataDescriptorPair& second) const {
      return DataDescriptorEquals()(&first.first, &second.first) &&
             DataDescriptorEquals()(&first.second, &second.second);
    }
  };

  std::unordered_map<DataDescriptorPair, std::vector<bool>,
                     DataDescriptorPairHash, DataDescriptorPairEquals>
      candidate_composite_synonyms;

  while (closure_computation_required) {
    closure_computation_required = false;
    for (auto representative :
         synonymous_.GetEquivalenceClassRepresentatives()) {
      auto equivalence_class = synonymous_.GetEquivalenceClass(*representative);
      for (auto dd1_it = equivalence_class.begin();
           dd1_it != equivalence_class.end(); ++dd1_it) {
        auto dd1 = *dd1_it;
        if (dd1->index_size() == 0) {
          continue;
        }
        auto dd2_it = dd1_it;
        for (++dd2_it; dd2_it != equivalence_class.end(); ++dd2_it) {
          auto dd2 = *dd2_it;
          if (dd2->index_size() == 0) {
            continue;
          }
          if (dd1->index(dd1->index_size() - 1) !=
              dd2->index(dd2->index_size() - 1)) {
            continue;
          }
          protobufs::DataDescriptor dd1_prefix;
          dd1_prefix.set_object(dd1->object());
          for (uint32_t i = 0; i < static_cast<uint32_t>(dd1->index_size() - 1);
               i++) {
            dd1_prefix.add_index(dd1->index(i));
          }
          protobufs::DataDescriptor dd2_prefix;
          dd2_prefix.set_object(dd2->object());
          for (uint32_t i = 0; i < static_cast<uint32_t>(dd2->index_size() - 1);
               i++) {
            dd2_prefix.add_index(dd2->index(i));
          }
          assert(!DataDescriptorEquals()(&dd1_prefix, &dd2_prefix) &&
                 "By construction these prefixes should be different.");
          if (synonymous_.Exists(dd1_prefix) &&
              synonymous_.Exists(dd2_prefix) &&
              synonymous_.IsEquivalent(dd1_prefix, dd2_prefix)) {
            continue;
          }

          auto dd1_root_type_id =
              context->get_def_use_mgr()->GetDef(dd1->object())->type_id();
          auto dd2_root_type_id =
              context->get_def_use_mgr()->GetDef(dd2->object())->type_id();
          auto dd1_penultimate_index_type =
              fuzzerutil::WalkCompositeTypeIndices(context, dd1_root_type_id,
                                                   dd1_prefix.index());
          auto dd2_penultimate_index_type =
              fuzzerutil::WalkCompositeTypeIndices(context, dd2_root_type_id,
                                                   dd2_prefix.index());
          if (dd1_penultimate_index_type != dd2_penultimate_index_type) {
            continue;
          }

          // At this point, we know we have synonymous data descriptors of the
          // form:
          //   root1[index_prefix1, last_index]
          //   root2[index_prefix2, last_index]
          // with the same last_index, such that root1[index_prefix1] and
          // root2[index_prefix2] have the same type.

          uint32_t num_elements_in_composite;
          auto composite_type =
              context->get_type_mgr()->GetType(dd1_penultimate_index_type);
          auto composite_type_instruction =
              context->get_def_use_mgr()->GetDef(dd1_penultimate_index_type);
          if (composite_type->AsArray()) {
            num_elements_in_composite =
                fuzzerutil::GetArraySize(*composite_type_instruction, context);
            if (num_elements_in_composite == 0) {
              // This indicates that the array has an unknown size, in which
              // case we cannot be sure we have matched all of its elements with
              // synonymous elements of another array.
              continue;
            }
          } else if (composite_type->AsMatrix()) {
            num_elements_in_composite =
                composite_type->AsMatrix()->element_count();
          } else if (composite_type->AsStruct()) {
            num_elements_in_composite = fuzzerutil::GetNumberOfStructMembers(
                *composite_type_instruction);
          } else {
            assert(composite_type->AsVector());
            num_elements_in_composite =
                composite_type->AsVector()->element_count();
          }

          DataDescriptorPair candidate_composite_synonym(dd1_prefix,
                                                         dd2_prefix);
          auto existing_entry =
              candidate_composite_synonyms.find(candidate_composite_synonym);
          if (existing_entry == candidate_composite_synonyms.end()) {
            std::vector<bool> entry;
            for (uint32_t i = 0; i < num_elements_in_composite; i++) {
              entry.push_back(i == dd1->index(dd1->index_size() - 1));
            }
            candidate_composite_synonyms[candidate_composite_synonym] = entry;
            existing_entry =
                candidate_composite_synonyms.find(candidate_composite_synonym);
          } else {
            existing_entry->second[dd1->index(dd1->index_size() - 1)] = true;
          }
          assert(existing_entry != candidate_composite_synonyms.end());
          bool all_components_match = true;
          for (uint32_t i = 0; i < num_elements_in_composite; i++) {
            if (!existing_entry->second[i]) {
              all_components_match = false;
              break;
            }
          }
          if (all_components_match) {
            // We do not recursively add the fact that dd1_prefix is synonymous
            // with dd2_prefix, as we have deduced this by *already* knowing
            // that their subcomponents are synonymous.
            synonymous_.MakeEquivalent(dd1_prefix, dd2_prefix);
            closure_computation_required = true;
            candidate_composite_synonyms.erase(candidate_composite_synonym);
          }
        }
      }
    }
  }
}

std::vector<const protobufs::DataDescriptor*>
FactManager::DataSynonymFacts::GetSynonymsForDataDescriptor(
    const protobufs::DataDescriptor& data_descriptor,
    opt::IRContext* context) const {
  const_cast<FactManager::DataSynonymFacts*>(this)->ComputeClosureOfFacts(
      context);
  if (synonymous_.Exists(data_descriptor)) {
    return synonymous_.GetEquivalenceClass(data_descriptor);
  }
  return std::vector<const protobufs::DataDescriptor*>();
}

std::vector<uint32_t>
FactManager::DataSynonymFacts ::GetIdsForWhichSynonymsAreKnown(
    opt::IRContext* context) const {
  const_cast<FactManager::DataSynonymFacts*>(this)->ComputeClosureOfFacts(
      context);
  std::vector<uint32_t> result;
  for (auto& data_descriptor : synonymous_.GetAllKnownValues()) {
    if (data_descriptor->index().empty()) {
      result.push_back(data_descriptor->object());
    }
  }
  return result;
}

bool FactManager::DataSynonymFacts::IsSynonymous(
    const protobufs::DataDescriptor& data_descriptor1,
    const protobufs::DataDescriptor& data_descriptor2,
    opt::IRContext* context) const {
  const_cast<FactManager::DataSynonymFacts*>(this)->ComputeClosureOfFacts(
      context);
  return synonymous_.Exists(data_descriptor1) &&
         synonymous_.Exists(data_descriptor2) &&
         synonymous_.IsEquivalent(data_descriptor1, data_descriptor2);
}

// End of data synonym facts
//==============================

FactManager::FactManager()
    : uniform_constant_facts_(MakeUnique<ConstantUniformFacts>()),
      data_synonym_facts_(MakeUnique<DataSynonymFacts>()) {}

FactManager::~FactManager() = default;

void FactManager::AddFacts(const MessageConsumer& message_consumer,
                           const protobufs::FactSequence& initial_facts,
                           opt::IRContext* context) {
  for (auto& fact : initial_facts.fact()) {
    if (!AddFact(fact, context)) {
      message_consumer(
          SPV_MSG_WARNING, nullptr, {},
          ("Invalid fact " + ToString(fact) + " ignored.").c_str());
    }
  }
}

bool FactManager::AddFact(const fuzz::protobufs::Fact& fact,
                          opt::IRContext* context) {
  switch (fact.fact_case()) {
    case protobufs::Fact::kConstantUniformFact:
      return uniform_constant_facts_->AddFact(fact.constant_uniform_fact(),
                                              context);
    case protobufs::Fact::kDataSynonymFact:
      data_synonym_facts_->AddFact(fact.data_synonym_fact(), context);
      return true;
    default:
      assert(false && "Unknown fact type.");
      return false;
  }
}

void FactManager::AddFactDataSynonym(const protobufs::DataDescriptor& data1,
                                     const protobufs::DataDescriptor& data2,
                                     opt::IRContext* context) {
  protobufs::FactDataSynonym fact;
  *fact.mutable_data1() = data1;
  *fact.mutable_data2() = data2;
  data_synonym_facts_->AddFact(fact, context);
}

std::vector<uint32_t> FactManager::GetConstantsAvailableFromUniformsForType(
    opt::IRContext* ir_context, uint32_t type_id) const {
  return uniform_constant_facts_->GetConstantsAvailableFromUniformsForType(
      ir_context, type_id);
}

const std::vector<protobufs::UniformBufferElementDescriptor>
FactManager::GetUniformDescriptorsForConstant(opt::IRContext* ir_context,
                                              uint32_t constant_id) const {
  return uniform_constant_facts_->GetUniformDescriptorsForConstant(ir_context,
                                                                   constant_id);
}

uint32_t FactManager::GetConstantFromUniformDescriptor(
    opt::IRContext* context,
    const protobufs::UniformBufferElementDescriptor& uniform_descriptor) const {
  return uniform_constant_facts_->GetConstantFromUniformDescriptor(
      context, uniform_descriptor);
}

std::vector<uint32_t> FactManager::GetTypesForWhichUniformValuesAreKnown()
    const {
  return uniform_constant_facts_->GetTypesForWhichUniformValuesAreKnown();
}

const std::vector<std::pair<protobufs::FactConstantUniform, uint32_t>>&
FactManager::GetConstantUniformFactsAndTypes() const {
  return uniform_constant_facts_->GetConstantUniformFactsAndTypes();
}

std::vector<uint32_t> FactManager::GetIdsForWhichSynonymsAreKnown(
    opt::IRContext* context) const {
  return data_synonym_facts_->GetIdsForWhichSynonymsAreKnown(context);
}

std::vector<const protobufs::DataDescriptor*>
FactManager::GetSynonymsForDataDescriptor(
    const protobufs::DataDescriptor& data_descriptor,
    opt::IRContext* context) const {
  return data_synonym_facts_->GetSynonymsForDataDescriptor(data_descriptor,
                                                           context);
}

std::vector<const protobufs::DataDescriptor*> FactManager::GetSynonymsForId(
    uint32_t id, opt::IRContext* context) const {
  return GetSynonymsForDataDescriptor(MakeDataDescriptor(id, {}), context);
}

bool FactManager::IsSynonymous(
    const protobufs::DataDescriptor& data_descriptor1,
    const protobufs::DataDescriptor& data_descriptor2,
    opt::IRContext* context) const {
  return data_synonym_facts_->IsSynonymous(data_descriptor1, data_descriptor2,
                                           context);
};

}  // namespace fuzz
}  // namespace spvtools
