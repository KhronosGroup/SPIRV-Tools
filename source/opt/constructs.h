// Copyright (c) 2016 Google Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

#ifndef LIBSPIRV_OPT_CONSTRUCTS_H_
#define LIBSPIRV_OPT_CONSTRUCTS_H_

#include <functional>
#include <vector>

#include "spirv-tools/libspirv.h"
#include "spirv/spirv.h"

namespace spvtools {
namespace opt {
namespace ir {

class Function;
class Module;

struct Payload {
  Payload(spv_operand_type_t t, std::vector<uint32_t>&& w)
      : type(t), words(std::move(w)) {}

  spv_operand_type_t type;
  std::vector<uint32_t> words;

  // TODO(antiagainst): create fields for literal number kind, width, etc.
};

class Inst {
 public:
  Inst() : opcode_(SpvOpNop), type_id_(0), result_id_(0) {}
  Inst(SpvOp op) : opcode_(op), type_id_(0), result_id_(0) {}
  Inst(const spv_parsed_instruction_t& inst);

  SpvOp opcode() const { return opcode_; }
  uint32_t type_id() const { return type_id_; }
  uint32_t result_id() const { return result_id_; }
  uint32_t GetSingleWordOperand(uint32_t index) const {
    return GetSingleWordPayload(index + TypeResultIdCount());
  }
  uint32_t NumOperandWords() const;
  const Payload& GetOperand(uint32_t index) const {
    return GetPayload(index + TypeResultIdCount());
  }
  uint32_t NumOperands() const {
    return static_cast<uint32_t>(payloads_.size() - TypeResultIdCount());
  }
  uint32_t GetSingleWordPayload(uint32_t index) const;
  uint32_t NumPayloadWords() const {
    return NumOperandWords() + TypeResultIdCount();
  }
  const Payload& GetPayload(uint32_t index) const;
  uint32_t NumPayloads() const {
    return static_cast<uint32_t>(payloads_.size());
  }
  void SetPayload(uint32_t index, std::vector<uint32_t>&& data);

  void ToBinary(std::vector<uint32_t>* binary) const;

 private:
  uint32_t TypeResultIdCount() const {
    return (type_id_ != 0) + (result_id_ != 0);
  }

  SpvOp opcode_;
  uint32_t type_id_;
  uint32_t result_id_;
  std::vector<Payload> payloads_;
};

class BasicBlock {
 public:
  BasicBlock(Function* function, Inst&& label)
      : function_(function), label_(label) {}

  void AddInstruction(Inst&& i) { insts_.push_back(std::move(i)); }

  void ForEachInst(const std::function<void(Inst*)>& f);

  void ToBinary(std::vector<uint32_t>* binary) const;

 private:
  Function* function_;
  Inst label_;
  std::vector<Inst> insts_;
};

class Function {
 public:
  Function(Module* module, Inst&& def_inst)
      : module_(module), def_inst_(def_inst), end_inst_(SpvOpFunctionEnd) {}

  void AddParameter(Inst&& p) { params_.push_back(std::move(p)); }
  void AddBasicBlock(BasicBlock&& b) { blocks_.push_back(std::move(b)); }

  void ForEachInst(const std::function<void(Inst*)>& f);

  void ToBinary(std::vector<uint32_t>* binary) const;

 private:
  Module* module_;
  Inst def_inst_;
  std::vector<Inst> params_;
  std::vector<BasicBlock> blocks_;
  Inst end_inst_;
};

struct ModuleHeader {
  uint32_t magic_number;
  uint32_t version;
  uint32_t generator;
  uint32_t bound;
  uint32_t reserved;
};

class Module {
 public:
  Module() : header_({}) {}

  void SetHeader(const ModuleHeader& header) { header_ = header; }
  void AddCapability(Inst&& c) { capabilities_.push_back(std::move(c)); }
  void AddExtension(Inst&& e) { extensions_.push_back(std::move(e)); }
  void AddExtInstSet(Inst&& e) { ext_inst_sets_.push_back(std::move(e)); }
  void SetMemoryModel(Inst&& m) { memory_model_ = m; }
  void AddEntryPoint(Inst&& e) { entry_points_.push_back(std::move(e)); }
  void AddExecutionMode(Inst&& e) { execution_modes_.push_back(std::move(e)); }
  void AddDebugInst(Inst&& d) { debugs_.push_back(std::move(d)); }
  void AddAnnotationInst(Inst&& a) { annotations_.push_back(std::move(a)); }
  void AddType(Inst&& t) { types_and_constants_.push_back(std::move(t)); }
  void AddConstant(Inst&& c) { types_and_constants_.push_back(std::move(c)); }
  void AddVariable(Inst&& v) { variables_.push_back(std::move(v)); }
  void AddFunction(Function&& f) { functions_.push_back(std::move(f)); }

  std::vector<Inst*> types();
  std::vector<Inst>& debugs() { return debugs_; }
  const std::vector<Inst>& annotations() const { return annotations_; }
  std::vector<Inst>& annotations() { return annotations_; }

  void ForEachInst(const std::function<void(Inst*)>& f);

  void ToBinary(std::vector<uint32_t>* binary) const;

 private:
  ModuleHeader header_;

  // The following fields respect the "Logical Layout of a Module" in
  // Section 2.4 of the SPIR-V specification.
  std::vector<Inst> capabilities_;
  std::vector<Inst> extensions_;
  std::vector<Inst> ext_inst_sets_;
  Inst memory_model_;
  std::vector<Inst> entry_points_;
  std::vector<Inst> execution_modes_;
  std::vector<Inst> debugs_;
  std::vector<Inst> annotations_;
  std::vector<Inst> types_and_constants_;
  std::vector<Inst> variables_;
  std::vector<Function> functions_;
};

}  // namespace ir
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_CONSTRUCTS_H_
