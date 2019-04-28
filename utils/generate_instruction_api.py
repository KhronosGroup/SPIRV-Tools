#!/usr/bin/env python
# Copyright (c) 2019 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates instruction API from SPIR-V JSON grammar."""

import collections
import json
import sys

enums = json.load(open(sys.argv[1]))['spv']['enum']
grammar = json.load(open(sys.argv[2]))
out_type = sys.argv[3]

extra = set([
  # Instructions we want to customize with hand-written code, e.g.
  #'MemberDecorate',
])

WordType = 'val', 'uint32_t'
WordPairType = 'val', 'std::pair<uint32_t, uint32_t>'
StringType = 'str', 'const char *'

# Map operand types to C++ types.
typemap = dict(
  IdResultType=WordType,
  IdResult=WordType,
  IdRef=WordType,
  IdScope=WordType,
  IdMemorySemantics=WordType,

  PairIdRefIdRef=WordPairType,
  PairLiteralIntegerIdRef=WordPairType,
  PairIdRefLiteralInteger=WordPairType,

  LiteralString=StringType,
  LiteralInteger=WordType,
  # TODO(fjhenigman): check these types
  LiteralExtInstInteger=WordType,
  LiteralContextDependentNumber=WordType,
  LiteralSpecConstantOpInteger=WordType,
)

for i in enums:
    name = i['Name']
    if i['Type'] == 'Value':
        typemap[name] = 'val', 'Spv' + name
    elif i['Type'] == 'Bit':
        typemap[name] = 'val', 'uint32_t'

template = dict(

hpp='''
{instruction_classes}

{aliases}

template<typename... Args>
struct Dispatch {{
  virtual ~Dispatch() {{}}
  virtual spv_result_t do_default(const Instruction&, Args...) {{ return SPV_SUCCESS; }}
  virtual spv_result_t do_missing(const Instruction&, Args...) {{ return SPV_UNSUPPORTED; }}

  {handlers}

  spv_result_t operator()(const Instruction *i, Args... args) {{
    switch(i->Opcode()) {{
      {dispatch_cases}
      default:;
    }}
    return do_missing(*i, args...);
  }}
}};
''',

cpp='''
std::shared_ptr<Instruction> Instruction::Make(const spv_parsed_instruction_t *inst) {{
  switch(inst->opcode) {{
    {make_cases}
  }}
  return nullptr;
}}
'''
)

# To augment generated code, you can write {classname} derived from
# {basename} by hand.
instruction_template = '''struct {basename} : public Instruction {{
  static constexpr SpvOp Opcode = SpvOp{name};
  {basename}(const spv_parsed_instruction_t *i) : Instruction(i) {{}}
  {getters}
}};'''

alias_template = '''using {classname} = {alias};'''

getter_template = '{type} Get{opname}() const {{ return get{getter}({pos}); }}'

handler_template = 'virtual spv_result_t do_{name}(const {classname} &i, Args... args) {{ return do_default(i, args...); }}'

dispatch_case_template = 'case SpvOp{name}: return do_{name}(*i->Get<{classname}>(), args...);'

make_case_template = 'case SpvOp{name}: return std::make_shared<{classname}>(inst);'

# Characters to strip out of operand names.
strip = {ord(i):None for i in "' "}

fix_op_names = {
  # (name, pos) : new_name
  ('CopyMemory', 3)                          : 'SourceMemoryAccess',
  ('CopyMemorySized', 4)                     : 'SourceMemoryAccess',
  ('ExtInst', 4)                             : 'Operands',
  ('FunctionCall', 3)                        : 'Arguments',
  ('SubgroupAvcImeSetDualReferenceINTEL', 4) : 'SearchWindowConfig',
  ('TypeFunction', 2)                        : 'ParameterTypes',
  ('TypeOpaque', 1)                          : 'TypeName',
  ('TypeStruct', 1)                          : 'Members',
}

# Fill out given template for each dict, join with newlines.
def fill(template, dicts):
    return '\n'.join(template.format(**i) for i in dicts)

# Given the json for an instruction, return a dictionary for use with our templates.
def make_inst(inst):
    name = inst['opname']
    assert name[:2] == 'Op'
    name = name[2:]
    classname = "I" + name
    basename = ("B" + name) if name in extra else classname
    ops = []
    for pos, op in enumerate(inst.get('operands', [])):
        opkind = op['kind']
        opname = op.get('name', opkind).translate(strip)
        opname = fix_op_names.get((name, pos), opname)
        getter, cpptype = typemap[opkind]
        returntype = cpptype
        if getter == 'val':
            getter = 'val<%s>' % cpptype
        if not opname.isalnum():
            # TODO(fjhenigman): Deal with this case more robustly.
            opname = 'OptionalImageOperands'
        quant = op.get('quantifier')
        if quant is None:
            pass
        elif quant == "*":
            getter = 'vec<%s>' % cpptype
            cpptype = 'std::vector<%s>' % cpptype
            returntype = 'const ' + cpptype
        elif quant == "?":
            pass
        else:
            assert 0, 'unexpected quantifier'
        ops.append(dict(opname=opname, getter=getter, type=returntype, pos=pos))

    return dict(
        opcode = int(inst['opcode']),
        name = name,
        basename = basename,
        classname = classname,
        ops = ops,
        getters = fill(getter_template, ops),
    )

# Instructions grouped by opcode.
grouped_inst = collections.defaultdict(list)
for i in grammar['instructions']:
    inst = make_inst(i)
    grouped_inst[inst['opcode']].append(inst)

# Order by opcode then by name.
class_inst = [] # first in opcode group, generate class for it
alias_inst = [] # remainder of group, generate aliases for them
for (opcode, group) in sorted(grouped_inst.items()):
    group = sorted(group, key=lambda x:x['name'])
    class_inst.append(group.pop(0))
    for inst in group:
        inst['alias'] = class_inst[-1]['classname']
        alias_inst.append(inst)

print("// THIS FILE IS GENERATED")
print(template[out_type].format(
    instruction_classes = fill(instruction_template  , class_inst),
    handlers            = fill(handler_template      , class_inst),
    dispatch_cases      = fill(dispatch_case_template, class_inst),
    make_cases          = fill(make_case_template    , class_inst),
    aliases             = fill(alias_template        , alias_inst),
))
