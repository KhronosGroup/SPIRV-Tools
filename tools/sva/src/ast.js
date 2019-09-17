// Copyright 2019 The Khronos Group Inc.
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

class Module {
  constructor() {
    this.instructions_ = [];
    this.next_id_ = 1;

    this.types_ = {};
    this.assigned_ids_ = {};
  }

  instructions() { return this.instructions_; }

  instruction(val) { return this.instructions_[val]; }

  addInstruction(inst) {
    this.instructions_.push(inst);

    // Record type information
    if (inst.name() === "OpTypeInt" || inst.name() === "OpTypeFloat") {
      this.types_[inst.operand(0).name()] = {
        type: inst.name() === "OpTypeInt" ? "int" : "float",
        width: inst.operand(1).value(),
        signed: inst.operands().length === 3 ? inst.operand(2).value() : 1
      };
    }

    // Record operand result id's
    inst.operands().forEach((op) => {
      if (op.rawValue() !== undefined && op.type() === "result_id") {
        this.next_id_ = Math.max(this.next_id_, op.rawValue() + 1);
      }
    });
  }

  getType(name) { return this.types_[name]; }

  getId(name) {
    if (this.assigned_ids_[name] !== undefined) {
      return this.assigned_ids_[name];
    }

    let next = this.next_id_;
    this.assigned_ids_[name] = next;

    this.next_id_ += 1;
    return next;
  }

  getIdBounds() { return this.next_id_ - 1; }
}

class Instruction {
  constructor(name, opcode, operands) {
    this.name_ = name;
    this.opcode_ = opcode;
    this.operands_ = operands;
  }

  name() { return this.name_; }

  opcode() { return this.opcode_; }

  operands() { return this.operands_; }

  operand(val) { return this.operands_[val]; }
}

class Operand {
  constructor(mod, name, type, value, params) {
    this.module_ = mod;
    this.name_ = name;
    this.type_ = type;
    this.value_ = value;
    this.params_ = params;
  }

  name() { return this.name_; }

  length() {
    // Get the value just to force it to be filled.
    this.value();

    if (this.type_ === "string") {
      return Math.ceil((this.value_.length + 1) / 4);
    }

    let size = 1;
    for (const param of this.params_) {
      size += param.length();
    }
    return size;
  }

  type() { return this.type_; }

  rawValue() { return this.value_; }

  value() {
    if (this.value_ === undefined) {
      this.value_ = this.module_.getId(this.name_);
    }
    return this.value_;
  }

  params() { return this.params_; }
}

export {
  Module,
  Instruction,
  Operand
};
