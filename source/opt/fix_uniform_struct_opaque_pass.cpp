// Copyright (c) 2021 The Khronos Group Inc.
// Copyright (c) 2021 Valve Corporation
// Copyright (c) 2021 LunarG Inc.
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

#include "source/opt/fix_uniform_struct_opaque_pass.h"

#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>

#include "ir_context.h"
#include "source/util/string_utils.h"
#include "type_manager.h"
#include "def_use_manager.h"
#include "constants.h"

namespace spvtools {
namespace opt {

namespace {

// Avoid unused variable warning/error on Linux
#ifndef NDEBUG
#define USE_ASSERT(x) assert(x)
#else
#define USE_ASSERT(x) ((void)(x))
#endif


// Get the member name instruction of index 'member_index' of a struct
// struct_id: OpTypeStruct
Instruction* GetMemberNameInst(IRContext* cxt, uint32_t struct_id, uint32_t member_index) {
	assert(cxt->get_def_use_mgr()->GetDef(struct_id)!=nullptr);
	assert(cxt->get_def_use_mgr()->GetDef(struct_id)->opcode()==SpvOpTypeStruct);
	Instruction*const member_name_inst = cxt->GetMemberName(struct_id,member_index);
	assert(member_name_inst!=nullptr && member_name_inst->opcode()==SpvOpMemberName);
	if (member_name_inst==nullptr || member_name_inst->opcode()!=SpvOpMemberName) return nullptr;     // Sanity check
	return member_name_inst;
}

// Retrieve the member name string
// inst_member_name: SpvOpMemberName
std::string GetMemberNameString(const Instruction* inst_member_name) {
	if (inst_member_name==nullptr) return {};                                             // Sanity check
	assert(inst_member_name->opcode()==SpvOpMemberName);
	return utils::MakeString(inst_member_name->GetOperand(2u).words);
}

// Get the member name string of index 'member_index' of a struct
// struct_id: OpTypeStruct
std::string GetMemberName(IRContext* cxt, uint32_t struct_id, uint32_t member_index) {
	return GetMemberNameString(GetMemberNameInst(cxt,struct_id,member_index));
}

// Return the member index of a access chain instruction 
// inst_accesschain: SpvOpAccessChain
uint32_t GetAccessChainIndex(IRContext* cxt,const Instruction& inst_accesschain,uint32_t access_index = 0u) {
	assert(inst_accesschain.opcode()==SpvOpAccessChain);
	uint32_t inst_cst_int_id = inst_accesschain.GetSingleWordInOperand(access_index+1u);

	Instruction*const inst_cst_int = cxt->get_def_use_mgr()->GetDef(inst_cst_int_id);
	assert(inst_cst_int!=nullptr && inst_cst_int->opcode()==SpvOpConstant);
	if (inst_cst_int==nullptr || inst_cst_int->opcode()!=SpvOpConstant) return ~0u;       // Sanity check

	const auto* index_constant = cxt->get_constant_mgr()->GetConstantFromInst(inst_cst_int);

	assert(index_constant!=nullptr);
	if (index_constant==nullptr) return ~0u;                                              // Sanity check

	// Get the sign-extended value, since access index is always treated as signed.
	const auto index_value = index_constant->GetSignExtendedValue();
	assert(index_value>=0);

	return index_value>=0 ? (uint32_t)index_value : ~0u;
}

// inst_accesschain: SpvOpAccessChain
uint32_t GetAccessChainCount(const Instruction& inst_accesschain) {
	assert(inst_accesschain.opcode()==SpvOpAccessChain);
	return inst_accesschain.NumInOperands()-1u;
}

// Get the type of the object pointed to
// ptr_id: OpTypePointer
uint32_t GetPtrTypeId(IRContext* cxt, uint32_t ptr_id) {
	Instruction*const inst_ptr = cxt->get_def_use_mgr()->GetDef(ptr_id);
	assert(inst_ptr!=nullptr && inst_ptr->opcode()==SpvOpTypePointer);
	if (inst_ptr==nullptr || inst_ptr->opcode()!=SpvOpTypePointer) return 0u;             // Sanity check
	return inst_ptr->GetSingleWordInOperand(1u);
}

// Set Uniform storage class on pointer instruction
// inst_ptr_id: SpvOpTypePointer
bool SetPointerStorageClassUniform(IRContext* cxt, uint32_t inst_ptr_id) {
	Instruction*const inst_ptr = cxt->get_def_use_mgr()->GetDef(inst_ptr_id);
	assert(inst_ptr!=nullptr && inst_ptr->opcode()==SpvOpTypePointer);
	if (inst_ptr==nullptr || inst_ptr->opcode()!=SpvOpTypePointer) return false;          // Sanity check

	inst_ptr->SetInOperand(0u,{(uint32_t)SpvStorageClassUniform});                        // Fix the storage class

	return true;
}

// Set Uniform storage class on whole branch of access chains
// Also fix loading of boolean (type mismatch on structure that contains opaque)
class FixVariableAccessChain
{
public:
	// Set Uniform storage class of variable and whole branch of access chains
	// Also fix loading of boolean in downstream access chain 
	// inst_variable: SpvOpVariable
	void process(IRContext* cxt,Instruction* inst_variable)
	{
		assert(inst_variable->opcode()==SpvOpVariable);
		inst_variable->SetInOperand(0u,{(uint32_t)SpvStorageClassUniform});  // Fix the storage class of the variable
		SetPointerStorageClassUniform(cxt,inst_variable->type_id());         // And also the pointer
		process_recursive_accesschain(cxt,inst_variable);                    // Recursive fix on access chain that use this variable
	}

protected:
	uint32_t bool_type_id_ = 0u;
	uint32_t u32_type_id_ = 0u;
	uint32_t ptr_uniform_u32_type_id_ = 0u;
	uint32_t cst_0u_id_ = 0u;


	void process_recursive_accesschain(IRContext* cxt,Instruction* inst)
	{
		cxt->get_def_use_mgr()->ForEachUser(
			inst,
			[this,cxt](Instruction* inst_user) {
				if (inst_user->opcode()==SpvOpAccessChain) {
					// Fix pointer storage class
					SetPointerStorageClassUniform(cxt,inst_user->type_id());

					// Fix loading of boolean/int mismatch
					uint32_t ptr_type_id = GetPtrTypeId(cxt,inst_user->type_id());
					assert(ptr_type_id!=0u);
					if (ptr_type_id==0u) return;                                // Sanity check
					Instruction*const ptr_type_inst = cxt->get_def_use_mgr()->GetDef(ptr_type_id);
					assert(ptr_type_inst!=nullptr);
					if (ptr_type_inst==nullptr) return;                         // Sanity check

					if (ptr_type_inst->opcode()==SpvOpTypeBool) {
						bool_type_id_ = ptr_type_id;
						fix_accesschain_bool(cxt,inst_user);
					}
					else if (ptr_type_inst->opcode()==SpvOpTypeStruct) {
						process_recursive_accesschain(cxt,inst_user);
					}
				}
			});
	}

	// %a = OpAccessChain %_ptr_Uniform_bool <base> <index>
	// %b = OpLoad %bool %a
	//   -->
	// %a = OpAccessChain %_ptr_Uniform_uint <base> <index>
	// %b = OpLoad %uint %a
	// %c = OpINotEqual %bool %b %uint_0
	void fix_accesschain_bool(IRContext* cxt,Instruction* accesschain_inst)
	{
		assert(accesschain_inst!=nullptr);
		assert(bool_type_id_!=0u);

		// Retrieve required types & constants if first call
		if (u32_type_id_==0u) {
			// First use, retrieve/create the vec4 type
			auto* const type_mgr = cxt->get_type_mgr();
			u32_type_id_ = type_mgr->GetUIntTypeId();
			ptr_uniform_u32_type_id_ = type_mgr->FindPointerToType(u32_type_id_,SpvStorageClassUniform);
			auto* const cst_mgr = cxt->get_constant_mgr();
			const auto*const cst_0u = cst_mgr->GetConstant(type_mgr->GetType(u32_type_id_), {0u});
			const Instruction*const cst_0u_inst = cst_mgr->GetDefiningInstruction(cst_0u);
			assert(cst_0u_inst!=nullptr);
			cst_0u_id_ = cst_0u_inst!=nullptr ?                                 // Sanity check
				cst_0u_inst->result_id() :
				0u;
		}
		assert(u32_type_id_!=0u);
		assert(ptr_uniform_u32_type_id_!=0u);
		assert(cst_0u_id_!=0u);

		// Replace OpAccessChain ptr bool result type by ptr u32
		accesschain_inst->SetResultType(ptr_uniform_u32_type_id_);

		cxt->get_def_use_mgr()->ForEachUser(
			accesschain_inst,
			[this,cxt](Instruction* load_inst) {
				assert(load_inst->opcode()==SpvOpLoad);
				if (load_inst->opcode()!=SpvOpLoad) return;                     // Sanity check

				// Replace SpvOpLoad bool result type by u32
				load_inst->SetResultType(u32_type_id_);
				
				// Create OpINotEqual instruction
				const uint32_t inotequal_id = cxt->TakeNextId();
				Instruction*const inotequal_inst = new Instruction(
					cxt,
					SpvOpINotEqual,
					bool_type_id_,
					inotequal_id,
					std::initializer_list<Operand>{
						{SPV_OPERAND_TYPE_ID, {load_inst->result_id()}},
						{SPV_OPERAND_TYPE_ID, {cst_0u_id_}}});
				inotequal_inst->InsertAfter(load_inst);

				cxt->get_def_use_mgr()->ForEachUse(
					load_inst,
					[inotequal_id](Instruction* load_user_inst, uint32_t logical_operand_index) {
						// Load users now use OpINotEqual
						load_user_inst->SetOperand(logical_operand_index,{inotequal_id});
					});
			});
	}
};

// Get the net valid binding index for a specific descriptor set
uint32_t GetNextBindingIndex(
	IRContext* cxt,
	uint32_t samplers_descriptor_set)
{
	uint32_t res_binding = 0u;
	for (Instruction& inst_global : cxt->module()->types_values()) {
		if (inst_global.opcode()==SpvOpVariable) {
			uint32_t cur_descriptor_set = ~0u;
			uint32_t cur_binding = ~0u;
			cxt->get_def_use_mgr()->ForEachUser(
				&inst_global,
				[&cur_descriptor_set,&cur_binding](Instruction* inst_global_user) {
					if (inst_global_user->opcode()==SpvOpDecorate) {
						switch (inst_global_user->GetSingleWordInOperand(1u)) {
							case SpvDecorationDescriptorSet: cur_descriptor_set = inst_global_user->GetSingleWordInOperand(2u); break;
							case SpvDecorationBinding:       cur_binding        = inst_global_user->GetSingleWordInOperand(2u); break;
						}
					}
				});
			assert((cur_descriptor_set!=~0u)==(cur_binding!=~0u));
			if (cur_descriptor_set==samplers_descriptor_set && cur_binding!=~0u) {
				// Use same descriptor set, retrieve maximum binding
				res_binding = std::max(res_binding,cur_binding+1u);
			}
		}
	}

	return res_binding;
}

size_t HashCombine(size_t hash0,size_t hash1) {
	hash0 ^= hash1 + 0x9e3779b9ull + (hash0 << 11ull) + (hash0 >> 21ull);
	return hash0;
}

class FlattenOpaqueVariables
{
public:
	FlattenOpaqueVariables(uint32_t samplers_descriptor_set) : samplers_descriptor_set_(samplers_descriptor_set) {}

	// Get/create the flatten variable that replace a whole access chain on opaque member type
	// inst_access_chain: OpTypeAccessChain
	// post: invalidate analysis types & decoration
	uint32_t fetch_variable_id(IRContext* cxt, const Instruction& inst_access_chain) {
		assert(inst_access_chain.opcode()==SpvOpAccessChain);
		
		AccessChainKey chain_key = generate_access_chain_key(cxt, inst_access_chain);

		if (chain_key.root_variable_id==0u) return 0u;                         // Early exit, not a chain on structures (may be arrays)

		auto generated_variable_id = generated_variable_ids.find(chain_key);
		if (generated_variable_id==generated_variable_ids.end()) {
			// Generate new flatten sampler
			const uint32_t binding_index = fetch_samplers_next_binding(cxt);
			const uint32_t inst_variable_id = cxt->TakeNextId();
			cxt->module()->AddGlobalValue(std::unique_ptr<Instruction>(new Instruction(
				cxt,
				SpvOpVariable,
				inst_access_chain.type_id(),
				inst_variable_id,
				std::initializer_list<Operand>{{
					SPV_OPERAND_TYPE_STORAGE_CLASS,
					{static_cast<uint32_t>(SpvStorageClassUniformConstant)}}})));
			cxt->module()->AddDebug2Inst(std::unique_ptr<Instruction>(new Instruction(
				cxt,
				SpvOpName,
				0u,              // no type id
				0u,              // no result id
				std::initializer_list<Operand>{
					{SPV_OPERAND_TYPE_ID, {inst_variable_id}},
					{SPV_OPERAND_TYPE_LITERAL_STRING, utils::MakeVector(chain_key.flatten_name)}})));
			cxt->module()->AddAnnotationInst(std::unique_ptr<Instruction>(new Instruction(
				cxt,
				SpvOpDecorate,
				0u,              // no type id
				0u,              // no result id
				std::initializer_list<Operand>{
					{SPV_OPERAND_TYPE_ID, {inst_variable_id}},
					{SPV_OPERAND_TYPE_DECORATION, {SpvDecorationDescriptorSet}},
					{SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER, {samplers_descriptor_set_}}})));
			cxt->module()->AddAnnotationInst(std::unique_ptr<Instruction>(new Instruction(
				cxt,
				SpvOpDecorate,
				0u,               // no type id
				0u,              // no result id
				std::initializer_list<Operand>{
					{SPV_OPERAND_TYPE_ID, {inst_variable_id}},
					{SPV_OPERAND_TYPE_DECORATION, {SpvDecorationBinding}},
					{SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER, {binding_index}}})));

			generated_variable_id = generated_variable_ids.emplace(std::move(chain_key),inst_variable_id).first;
		}

		return generated_variable_id->second;
	}

	// Fix storage class of whole struct chains that contained opaque type members
	void fix_storage_class(IRContext* cxt) {
		std::vector<uint32_t> root_variable_ids;
		root_variable_ids.reserve(generated_variable_ids.size());
		for (const auto& var_id : generated_variable_ids) {
			root_variable_ids.push_back(var_id.first.root_variable_id);
		}
		std::sort(root_variable_ids.begin(),root_variable_ids.end());
		root_variable_ids.erase(
			std::unique(root_variable_ids.begin(),root_variable_ids.end()),
			root_variable_ids.end());

		// Note: should have only one variable: GL default uniform block
		FixVariableAccessChain fix_var_accesschain;
		for (uint32_t root_variable_id : root_variable_ids) {
			Instruction*const inst_root_variable = cxt->get_def_use_mgr()->GetDef(root_variable_id);
			assert(inst_root_variable!=nullptr);
			if (inst_root_variable==nullptr) continue;                      // Sanity check
			fix_var_accesschain.process(cxt, inst_root_variable);
		}
	}

protected:
	struct AccessChainKey {
		uint32_t root_variable_id = 0u;                                     // SpvOpVariable
		std::string flatten_name;                                           // Flatten sampler name
	};

	struct AccessChainKeyHash {
		size_t operator()(const AccessChainKey& v) const {
			return HashCombine(std::hash<std::string>()(v.flatten_name),v.root_variable_id);
		}
	};

	struct AccessChainKeyEqual {
		bool operator()(const AccessChainKey& a,const AccessChainKey& b) const {
			return a.flatten_name==b.flatten_name && a.root_variable_id==b.root_variable_id;
		}
	};

	using VariableIdsPerAccessChain = std::unordered_map<
		AccessChainKey,
		uint32_t,
		AccessChainKeyHash,
		AccessChainKeyEqual>;

	const uint32_t samplers_descriptor_set_;
	uint32_t samplers_next_binding_ = ~0u;                                  // Lazy retrieve
	VariableIdsPerAccessChain generated_variable_ids;                       // Generated variable IDs for each extracted sampler 


	AccessChainKey generate_access_chain_key(IRContext* cxt, const Instruction& inst) {
		if (inst.opcode()==SpvOpVariable) {
			AccessChainKey k;
			k.root_variable_id = inst.result_id();
			return k;
		}
		else if (inst.opcode()==SpvOpAccessChain) {
			const Instruction*const parent_inst = cxt->get_def_use_mgr()->GetDef(
				inst.GetSingleWordInOperand(0u));                                   // 0: <id> Base
			assert(parent_inst!=nullptr);
			if (parent_inst==nullptr) return {};                                    // Sanity check

			AccessChainKey chain_key = generate_access_chain_key(cxt, *parent_inst);

			if (chain_key.root_variable_id==0u) return {};

			uint32_t struct_id = GetPtrTypeId(cxt, parent_inst->type_id());

			const uint32_t access_chain_count = GetAccessChainCount(inst);
			for (uint32_t access_index = 0u; access_index<access_chain_count; ++access_index) {
				assert(struct_id!=0u);
				if (struct_id==0u) return {};                                       // Sanity check

				const Instruction*const struct_inst = cxt->get_def_use_mgr()->GetDef(struct_id);
				assert(struct_inst!=nullptr);
				if (struct_inst==nullptr) return {};                                // Sanity check

				assert(struct_inst->opcode()==SpvOpTypeStruct || (access_chain_count==1u && parent_inst->opcode()==SpvOpVariable));
				if (struct_inst->opcode()!=SpvOpTypeStruct) return {};              // Not a struct may be an array

				const uint32_t member_index = GetAccessChainIndex(cxt, inst, access_index);
				assert(member_index<struct_inst->NumInOperands());
				if (member_index>=struct_inst->NumInOperands()) return {};          // Sanity check

				if (!chain_key.flatten_name.empty()) {
					chain_key.flatten_name.push_back('.');
				}

				const std::string member_name = GetMemberName(cxt, struct_id, member_index);
				assert(!member_name.empty());
				chain_key.flatten_name.append(member_name);

				// For next access chain index (not used if last one)
				struct_id = struct_inst->GetSingleWordInOperand(member_index);      // 0..n: <id>,<id>... Member types
			}

			assert(struct_id==GetPtrTypeId(cxt,inst.type_id()));

			return chain_key;
		}

		// Access chain parent type not supported
		assert(false);
		return {};
	}

	uint32_t fetch_samplers_next_binding(IRContext* cxt) {
		if (samplers_next_binding_==~0u) {
			// First use: retrieve next binding index
			// Must be done before adding any instruction to avoid definition manager related issue
			samplers_next_binding_ = GetNextBindingIndex(cxt, samplers_descriptor_set_);
		}
		assert(samplers_next_binding_!=~0u);
		return samplers_next_binding_++;
	}
};

}  // namespace


Pass::Status FixUniformStructOpaquePass::Process() {
	bool changed = false;
	InstructionList to_kill;                   // Delayed destruction to preserve def manager comparisions
	FlattenOpaqueVariables flatten_opaque_variables(samplers_descriptor_set_);

	// Retrieve and process access chain that returns pointer on opaque type
	for (Instruction& inst_type : get_module()->types_values()) {
		if (inst_type.opcode()==SpvOpTypeSampledImage) {                                            // TODO: TypeSampler TypeImage
			context()->get_def_use_mgr()->ForEachUser(
				&inst_type,
				[&](Instruction* inst_opaque_type_user) {
					if (inst_opaque_type_user->opcode()==SpvOpLoad) {
						assert(inst_opaque_type_user->type_id()==inst_type.result_id());

						Instruction* const inst_load_src = context()->get_def_use_mgr()->GetDef(
							inst_opaque_type_user->GetSingleWordInOperand(0u));                                  // OpLoad 0: <id> Pointer
						assert(inst_load_src!=nullptr);
						if (inst_load_src==nullptr) return;                                         // Sanity check

						if (inst_load_src->opcode()==SpvOpAccessChain) {
							// Invalid access chain, kill it and load directly a flatten sampler variable
							uint32_t inst_sampler_id = flatten_opaque_variables.fetch_variable_id(context(),*inst_load_src);
							
							if (inst_sampler_id!=0u) {
								changed = true;

								inst_opaque_type_user->SetInOperand(0u, {inst_sampler_id});         // OpLoad 0: <id> Pointer

								inst_load_src->RemoveFromList();
								to_kill.push_back(std::unique_ptr<Instruction>(inst_load_src));
							}
						}
					}
				});
		}
	}

	uint32_t u32_type_id = 0u;                                            // Lazy creation

	// Retrieve and process opaque type struct members
	for (Instruction& inst : get_module()->types_values()) {
		if (inst.opcode()==SpvOpTypeStruct) {
			const uint32_t num_in_operands = inst.NumInOperands();

			for (uint32_t in_operand_index = 0u; in_operand_index<num_in_operands; ++in_operand_index) {
				assert(inst.GetInOperand(in_operand_index).type==SPV_OPERAND_TYPE_ID);
				const uint32_t member_id = inst.GetSingleWordInOperand(in_operand_index);
				const Instruction*const member_inst = get_def_use_mgr()->GetDef(member_id);
				assert(member_inst!=nullptr);
				if (member_inst==nullptr) continue;                       // Sanity check

				// Iterate on struct members
				if (member_inst->opcode()==SpvOpTypeSampledImage ||
					member_inst->opcode()==SpvOpTypeSampler ||
					member_inst->opcode()==SpvOpTypeImage)
				{
					// This structure has an opaque type member
					changed = true;

					// Replace by u32 (match default type set by glslang, deduced from offsets)
					if (u32_type_id==0u) {
						// First use, retrieve/create the vec4 type
						u32_type_id = context()->get_type_mgr()->GetUIntTypeId();
					}

					assert(u32_type_id!=0u);

					inst.SetInOperand(in_operand_index, {u32_type_id});

					Instruction*const member_name_inst = GetMemberNameInst(context(),inst.result_id(),in_operand_index);
					assert(member_name_inst!=nullptr);
					if (member_name_inst==nullptr) continue;              // Sanity check

					// Remove member name to avoid conflicts at reflexion
					member_name_inst->RemoveFromList();
					to_kill.push_back(std::unique_ptr<Instruction>(member_name_inst));
				}
			}
		}
	}

	if (changed) {
		context()->InvalidateAnalyses(IRContext::kAnalysisDefUse);        // Update def uses, required for the storage fix pass just below

		flatten_opaque_variables.fix_storage_class(context());

		context()->InvalidateAnalyses((IRContext::Analysis)(
			IRContext::kAnalysisTypes|
			IRContext::kAnalysisConstants|
			IRContext::kAnalysisCFG|
			IRContext::kAnalysisDecorations|
			IRContext::kAnalysisDefUse|
			IRContext::kAnalysisInstrToBlockMapping));
	}

	return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
