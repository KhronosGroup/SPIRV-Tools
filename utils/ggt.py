#!/usr/bin/env python3
# Copyright (c) 2016 Google Inc.
#
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
"""Generates compressed grammar tables from SPIR-V JSON grammar."""

# Note: This will eventually replace generate_grammar_tables.py

import errno
import json
import os.path
import re
import sys
from typing import Dict, List, Tuple

# Find modules relative to the directory containing this script.
# This is needed for hermetic Bazel builds, where the Table files are bundled
# together with this script, while keeping their relative locations.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Table.Context import Context
from Table.IndexRange import IndexRange
from Table.Operand import Operand

# Extensions to recognize, but which don't necessarily come from the SPIR-V
# core or KHR grammar files.  Get this list from the SPIR-V registry web page.
# NOTE: Only put things on this list if it is not in those grammar files.
EXTENSIONS_FROM_SPIRV_REGISTRY_AND_NOT_FROM_GRAMMARS = """
SPV_AMD_gcn_shader
SPV_AMD_gpu_shader_half_float
SPV_AMD_gpu_shader_int16
SPV_AMD_shader_trinary_minmax
SPV_KHR_non_semantic_info
SPV_EXT_relaxed_printf_string_address_space
"""

MODE='new'

def convert_min_required_version(version): # (version: str | None) -> str
    """Converts the minimal required SPIR-V version encoded in the grammar to
    the symbol in SPIRV-Tools."""
    if version is None:
        return 'SPV_SPIRV_VERSION_WORD(1, 0)'
    if version == 'None':
        return '0xffffffffu'
    return 'SPV_SPIRV_VERSION_WORD({})'.format(version.replace('.', ','))


def convert_max_required_version(version): # (version: str | None) -> str
    """Converts the maximum required SPIR-V version encoded in the grammar to
    the symbol in SPIRV-Tools."""
    if version is None:
        return '0xffffffffu'
    return 'SPV_SPIRV_VERSION_WORD({})'.format(version.replace('.', ','))


def c_bool(b: bool) -> str:
    return 'true' if b else 'false'


def ctype(kind: str, quantifier: str) -> str:
    """Returns the corresponding operand type used in spirv-tools for the given
    operand kind and quantifier used in the JSON grammar.

    Arguments:
      - kind, e.g. 'IdRef'
      - quantifier, e.g. '', '?', '*'

    Returns:
      a string of the enumerant name in spv_operand_type_t
    """
    if kind == '':
        raise Error("operand JSON object missing a 'kind' field")
    # The following cases are where we differ between the JSON grammar and
    # spirv-tools.
    if kind == 'IdResultType':
        kind = 'TypeId'
    elif kind == 'IdResult':
        kind = 'ResultId'
    elif kind == 'IdMemorySemantics' or kind == 'MemorySemantics':
        kind = 'MemorySemanticsId'
    elif kind == 'IdScope' or kind == 'Scope':
        kind = 'ScopeId'
    elif kind == 'IdRef':
        kind = 'Id'

    elif kind == 'ImageOperands':
        kind = 'Image'
    elif kind == 'Dim':
        kind = 'Dimensionality'
    elif kind == 'ImageFormat':
        kind = 'SamplerImageFormat'
    elif kind == 'KernelEnqueueFlags':
        kind = 'KernelEnqFlags'

    elif kind == 'LiteralExtInstInteger':
        kind = 'ExtensionInstructionNumber'
    elif kind == 'LiteralSpecConstantOpInteger':
        kind = 'SpecConstantOpNumber'
    elif kind == 'LiteralContextDependentNumber':
        kind = 'TypedLiteralNumber'

    elif kind == 'PairLiteralIntegerIdRef':
        kind = 'LiteralIntegerId'
    elif kind == 'PairIdRefLiteralInteger':
        kind = 'IdLiteralInteger'
    elif kind == 'PairIdRefIdRef':  # Used by OpPhi in the grammar
        kind = 'Id'

    if kind == 'FPRoundingMode':
        kind = 'FpRoundingMode'
    elif kind == 'FPFastMathMode':
        kind = 'FpFastMathMode'

    if quantifier == '?':
        kind = 'Optional{}'.format(kind)
    elif quantifier == '*':
        kind = 'Variable{}'.format(kind)

    return 'SPV_OPERAND_TYPE_{}'.format(
        re.sub(r'([a-z])([A-Z])', r'\1_\2', kind).upper())


def convert_operand_kind(obj: Dict[str, str]) -> str:
    """Returns the corresponding operand type used in spirv-tools for the given
    operand kind and quantifier used in the JSON grammar.

    Arguments:
      - obj: an instruction operand, having keys:
          - 'kind', e.g. 'IdRef'
          - optionally, a quantifier: '?' or '*'

    Returns:
      a string of the enumerant name in spv_operand_type_t
    """
    kind = obj.get('kind', '')
    quantifier = obj.get('quantifier', '')
    return ctype(kind, quantifier)


def to_safe_identifier(s: str) -> str:
    """
    Returns a new string with all non-letters converted to underscores,
    and prepending 'k'.
    The result should be safe to use as a C identifier.
    """
    return 'k' + re.sub(r'[^a-zA-Z]', '_', s)


class Grammar():
    """
    Accumulates string and enum tables.
    The extensions and operand kinds lists are fixed at creation time.
    Prints tables for instructions, operand kinds, and underlying string
    and enum tables.
    Assumes an index range is emitted by printing an IndexRange object.
    """
    def __init__(self, extensions: List[str], operand_kinds:List[dict], printing_classes: List[str]) -> None:
        self.context = Context()
        self.extensions = extensions
        self.operand_kinds = sorted(operand_kinds, key = lambda ok: convert_operand_kind(ok))
        self.printing_classes = sorted([to_safe_identifier(x) for x in printing_classes])

        # The self.header_ignore_decls are only used to debug the flow.
        # They are copied into the C++ source code where they are more likely
        # to be seen by humans.
        self.header_ignore_decls: List[str] = [self.IndexRangeDecls()]

        # The self.header_decls content goes into core_tables_header.inc to be
        # included in a .h file.
        self.header_decls: List[str] = self.PrintingClassDecls()
        # The self.body_decls content goes into core_tables_body.inc to be included
        # in a .cpp file.  It includes definitions of static variables and
        # hidden functions.
        self.body_decls: List[str] = []

        if len(self.operand_kinds) == 0:
            raise Exception("operand_kinds should be a non-empty list")
        if len(self.extensions) == 0:
            raise Exception("extensions should be a non-empty list")

        # Preload the string table
        self.context.AddStringList('extension', extensions)

        # These operand kinds need to have their optional counterpart to also
        # be represented in the lookup tables, with the same content.
        self.operand_kinds_needing_optional_variant = [
                'ImageOperands',
                'AccessQualifier',
                'MemoryAccess',
                'PackedVectorFormat',
                'CooperativeMatrixOperands',
                'MatrixMultiplyAccumulateOperands',
                'RawAccessChainOperands',
                'FPEncoding']

    def dump(self) -> None:
        self.context.dump()

    def IndexRangeDecls(self) -> str:
        return """
struct IndexRange {
  uint32_t first = 0; // index of the first element in the range
  uint32_t count = 0; // number of elements in the range
};
constexpr inline IndexRange IR(uint32_t first, uint32_t count) {
  return {first, count};
}
"""

    def PrintingClassDecls(self) -> str:
        parts: List[str] = []
        parts.append("enum class PrintingClass : int {");
        parts.extend(["  {},".format(x) for x in self.printing_classes])
        parts.append("};\n")
        return parts

    def ExtensionEnumList(self) -> str:
        """
        Returns the spvtools::Extension enum values, as a string.
        This is kept separate because it will be included in 'source/extensions.h'
        which has an otherwise narrow interface.
        """
        return ',\n'.join(['k' + e for e in self.extensions])

    def ComputeOperandTables(self) -> None:
        """
        Returns the string for the C definitions of the operand kind tables.

        An operand kind such as ImageOperands also has an associated
        operand kind that is an 'optional' variant.
        These are represented as two distinct operand kinds in spv_operand_type_t.
        For example, ImageOperands maps to both SPV_OPERAND_TYPE_IMAGE, and also
        to SPV_OPERAND_TYPE_OPTIONAL_IMAGE.

        The definitions are:
         - kOperandsByValue: a 1-dimensional array of all operand descriptions
           sorted first by operand kind, then by operand value.
           Only non-optional operand kinds are represented here.

         - kOperandsByValueRangeByKind: a function mapping from operand kind to
           the index range into kOperandByValue.
           This has mappings for both concrete and corresponding optional operand kinds.

         - kOperandNames: a 1-dimensional array of all operand name-value pairs,
           sorted first by operand kinds, then by operand name.
           This can have more entries than the by-value array, because names
           can have string aliases. For example,the MemorySemantics value 0
           is named both "Relaxed" and "None".
           Each entry is represented by an index range into the string table.
           Only non-optional operand kinds are represented here.

         - kOperandNamesRangeByKind: a mapping from operand kind to the index
           range into kOperandNames.
           This has mappings for both concrete and corresponding optional operand kinds.

        """

        self.header_ignore_decls.append(
"""
struct NameValue {
  // Location of the null-terminated name in the global string table.
  IndexRange name;
  // Enum value in the binary format.
  uint32_t value;
};
// Describes a SPIR-V operand.
struct OperandDesc {
  uint32_t value;
  IndexRange operands_range;      // Indexes kOperandSpans
  IndexRange name_range;          // Indexes kStrings
  IndexRange aliases_range;       // Indexes kAliasSpans
  IndexRange capabilities_range;  // Indexes kCapabilitySpans
  // A set of extensions that enable this feature. If empty then this operand
  // value is in core and its availability is subject to minVersion. The
  // assembler, binary parser, and disassembler ignore this rule, so you can
  // freely process invalid modules.
  IndexRange extensions_range;  // Indexes kExtensionSpans
  // Minimal core SPIR-V version required for this feature, if without
  // extensions. ~0u means reserved for future use. ~0u and non-empty
  // extension lists means only available in extensions.
  uint32_t minVersion;
  uint32_t lastVersion;
  utils::Span<spv_operand_type_t> operands() const;
  utils::Span<char> name() const;
  utils::Span<IndexRange> aliases() const;
  utils::Span<spv::Capability> capabilities() const;
  utils::Span<spvtools::Extension> extensions() const;
  OperandDesc(const OperandDesc&) = delete;
  OperandDesc(OperandDesc&&) = delete;
};
""")

        def ShouldEmit(operand_kind_json: Dict[str,any]):
            """ Returns true if we should emit a table for the given
            operand kind.
            """
            category = operand_kind_json.get('category')
            return category in ['ValueEnum', 'BitEnum']

        # Populate kOperandNames
        operand_names: List[Tuple[IndexRange,int]] = []
        name_range_for_kind: Dict[str,IndexRange] = {}
        for operand_kind_json in self.operand_kinds:
            kind_key: str = convert_operand_kind(operand_kind_json)
            if ShouldEmit(operand_kind_json):
                operands = [Operand(o) for o in operand_kind_json['enumerants']]
                tuples: List[Tuple[str,int,str]] = []
                for o in operands:
                    tuples.append((o.enumerant, o.value, kind_key))
                    for a in o.aliases:
                        tuples.append((a, o.value, kind_key))
                tuples = sorted(tuples, key = lambda t: t[0])
                ir_tuples = [(self.context.AddString(t[0]),t[1],t[2]) for t in tuples]
                name_range_for_kind[kind_key] = IndexRange(len(operand_names), len(ir_tuples))
                operand_names.extend(ir_tuples)
            else:
                pass
        operand_name_strings: List[str] = []
        for i in range(0, len(operand_names)):
            ir, value, kind_key = operand_names[i]
            operand_name_strings.append('{{{}, {}}}, // {} {} in {}'.format(
                str(ir),value,i,self.context.GetString(ir),kind_key))

        parts: List[str] = []
        parts.append("""// Operand names and values, ordered by (operand kind, name)
// The fields in order are:
//   name, either the primary name or an alias, indexing into kStrings
//   enum value""")
        parts.append("std::array<NameValue, {}> kOperandNames{{{{".format(len(operand_name_strings)))
        parts.extend(['  ' + str(x) for x in operand_name_strings])
        parts.append("}};\n")
        self.body_decls.extend(parts)

        parts.append("""// Maps an operand kind to possible names for operands of that kind.
// The result is an IndexRange into kOperandNames, and the names
// are sorted by name within that span.
// An optional variant of a kind maps to the details for the corresponding
// concrete operand kind.""")
        parts = ["IndexRange OperandNameRangeForKind(spv_operand_type_t type) {\n  switch(type) {"]
        for kind_key, ir in name_range_for_kind.items():
            parts.append("    case {}: return {};".format(
                kind_key,
                str(name_range_for_kind[kind_key])))
        for kind in self.operand_kinds_needing_optional_variant:
            parts.append("    case {}: return {};".format(
                ctype(kind, '?'),
                str(name_range_for_kind[ctype(kind,'')])))
        parts.append("    default: break;");
        parts.append("  }\n  return IR(0,0);\n}\n")
        self.body_decls.extend(parts)

        # Populate kOperandsByValue
        operands_by_value: List[str] = []
        operands_by_value_by_kind: Dict[str,IndexRange] = {}
        for operand_kind_json in self.operand_kinds:
            kind_key: str = convert_operand_kind(operand_kind_json)
            if ShouldEmit(operand_kind_json):
                operands = [Operand(o) for o in operand_kind_json['enumerants']]
                operand_descs: List[str] = []
                for o in sorted(operands, key = lambda o: o.value):
                    suboperands = [convert_operand_kind(p) for p in o.parameters]
                    desc = [
                        o.value,
                        self.context.AddStringList('operand', suboperands),
                        str(self.context.AddString(o.enumerant)) + '/* {} */'.format(o.enumerant),
                        self.context.AddStringList('alias', o.aliases),
                        self.context.AddStringList('capability', o.capabilities),
                        self.context.AddStringList('extension', o.extensions),
                        convert_min_required_version(o.version),
                        convert_max_required_version(o.lastVersion),
                    ]
                    operand_descs.append('{' + ','.join([str(d) for d in desc]) + '}}, // {}'.format(kind_key))
                operands_by_value_by_kind[kind_key] = IndexRange(len(operands_by_value), len(operand_descs))
                operands_by_value.extend(operand_descs)
            else:
                pass

        parts = []
        parts.append("""// Operand descriptions, ordered by (operand kind, operand enum value).
// The fields in order are:
//   enum value
//   operands, an IndexRange into kOperandSpans
//   name, a character-counting IndexRange into kStrings
//   aliases, an IndexRange into kAliasSpans
//   capabilities, an IndexRange into kCapabilitySpans
//   extensions, as an IndexRange into kExtensionSpans
//   version, first version of SPIR-V that has it
//   lastVersion, last version of SPIR-V that has it""")
        parts.append("std::array<OperandDesc, {}> kOperandsByValue{{{{".format(len(operands_by_value)))
        parts.extend(['  ' + str(x) for x in operands_by_value])
        parts.append("}};\n")
        self.body_decls.extend(parts)

        parts = []
        parts.append("""// Maps an operand kind to possible operands for that kind.
// The result is an IndexRange into kOperandsByValue, and the operands
// are sorted by value within that span.
// An optional variant of a kind maps to the details for the corresponding
// concrete operand kind.""")
        parts.append("IndexRange OperandByValueRangeForKind(spv_operand_type_t type) {\n  switch(type) {")
        for kind_key, ir in operands_by_value_by_kind.items():
            parts.append("    case {}: return {};".format(
                kind_key,
                str(operands_by_value_by_kind[kind_key])))
        for kind in self.operand_kinds_needing_optional_variant:
            parts.append("    case {}: return {};".format(
                ctype(kind, '?'),
                str(operands_by_value_by_kind[ctype(kind,'')])))
        parts.append("    default: break;");
        parts.append("  }\n  return IR(0,0);\n}\n")
        self.body_decls.extend(parts)


    def ComputeInstructionTables(self, insts) -> None:
        """
        Creates declarations for instruction tables.
        Populates self.header_ignore_decls, self.body_decls.

        Params:
            insts: an array of instructions objects using the JSON schema
        """
        self.header_ignore_decls.append(
"""
// Describes an Instruction
struct InstructionDesc {
  const spv::Op value;
  const bool hasResult;
  const bool hasType;
  const IndexRange operands_range;      // Indexes kOperandSpans
  const IndexRange name_range;          // Indexes kStrings
  const IndexRange aliases_range;       // Indexes kAliasSpans
  const IndexRange capabilities_range;  // Indexes kCapbilitySpans
  // A set of extensions that enable this feature. If empty then this operand
  // value is in core and its availability is subject to minVersion. The
  // assembler, binary parser, and disassembler ignore this rule, so you can
  // freely process invalid modules.
  const IndexRange extensions_range;    // Indexes kExtensionSpans
  // Minimal core SPIR-V version required for this feature, if without
  // extensions. ~0u means reserved for future use. ~0u and non-empty
  // extension lists means only available in extensions.
  uint32_t minVersion;
  uint32_t lastVersion;
  PrintingClass printingClass; // Section of SPIR-V spec. e.g. kComposite, kImage
  utils::Span<spv_operand_type_t> operands() const;
  utils::Span<char> name() const;
  utils::Span<IndexRange> aliases() const;
  utils::Span<spv::Capability> capabilities() const;
  utils::Span<spvtools::Extension> extensions() const;
  OperandDesc(const OperandDesc&) = delete;
  OperandDesc(OperandDesc&&) = delete;
};
""")

        # Create the sorted list of opcode strings, without the 'Op' prefix.
        opcode_name_entries: List[str] = []
        name_value_pairs: List[Tuple[str,int]] = []
        for i in insts:
            name_value_pairs.append((i['opname'][2:], i['opcode']))
            for a in i.get('aliases',[]):
                name_value_pairs.append((a[2:], i['opcode']))
        name_value_pairs = sorted(name_value_pairs)
        inst_name_strings: List[str] = []
        for i in range(0, len(name_value_pairs)):
            name, value = name_value_pairs[i]
            ir = self.context.AddString(name)
            inst_name_strings.append('{{{}, {}}}, // {} {}'.format(str(ir),value,i,name))
        parts: List[str] = []
        parts.append("""// Opcode strings (without the 'Op' prefix) and opcode values, ordered by name.
// The fields in order are:
//   name, either the primary name or an alias, indexing into kStrings
//   opcode value""")
        parts.append("std::array<NameValue, {}> kInstructionNames{{{{".format(len(inst_name_strings)))
        parts.extend(['  ' + str(x) for x in inst_name_strings])
        parts.append("}};\n")
        self.body_decls.extend(parts)

        # Create the array of InstructionDesc
        lines: List[str] = []
        for inst in insts:
            parts: List[str] = []

            opname: str = inst['opname']

            operand_kinds = [convert_operand_kind(o) for o in inst.get('operands',[])]
            if opname == 'OpExtInst' and operand_kinds[-1] == 'SPV_OPERAND_TYPE_VARIABLE_ID':
                # The published grammar uses 'sequence of ID' at the
                # end of the ExtInst operands. But SPIRV-Tools uses
                # a specific pattern based on the particular opcode.
                # Drop it here.
                # See https://github.com/KhronosGroup/SPIRV-Tools/issues/233
                operand_kinds.pop()

            hasResult = 'SPV_OPERAND_TYPE_RESULT_ID' in operand_kinds
            hasType = 'SPV_OPERAND_TYPE_TYPE_ID' in operand_kinds

            # Remove the "Op" prefix from opcode alias names
            aliases = [name[2:] for name in inst.get('aliases',[])]

            parts.extend([
                'spv::Op::' + opname,
                c_bool(hasResult),
                c_bool(hasType),
                self.context.AddStringList('operand', operand_kinds),
                self.context.AddString(opname[2:]),
                self.context.AddStringList('alias', aliases),
                self.context.AddStringList('capability', inst.get('capabilities',[])),
                self.context.AddStringList('extension', inst.get('extensions',[])),
                convert_min_required_version(inst.get('version', None)),
                convert_max_required_version(inst.get('lastVersion', None)),
                'PrintingClass::' + to_safe_identifier(inst.get('class','@exclude'))
            ])

            lines.append('{{{}}},'.format(', '.join([str(x) for x in parts])))
        parts = []
        parts.append("""// Instruction descriptions, ordered by opcode.
// The fields in order are:
//   opcode
//   a boolean indicating if the instruction produces a result ID
//   a boolean indicating if the instruction result ID has a type
//   operands, an IndexRange into kOperandSpans
//   opcode name (without the 'Op' prefix), a character-counting IndexRange into kStrings
//   aliases, an IndexRange into kAliasSpans
//   capabilities, an IndexRange into kCapabilitySpans
//   extensions, as an IndexRange into kExtensionSpans
//   version, first version of SPIR-V that has it
//   lastVersion, last version of SPIR-V that has it""")
        parts.append("std::array<InstructionDesc, {}> kInstructionDesc{{{{".format(len(lines)));
        parts.extend(['  ' + l for l in lines])
        parts.append("}};\n");
        self.body_decls.extend(parts)


    def ComputeLeafTables(self) -> None:
        """
        Generates the tables that the instruction and operand tables point to.
        The tables are:
            - the string table
            - the table of sequences of:
               - capabilities
               - extensions
               - operands

        This method must be called after computing instruction and operand tables.
        """

        def c_str(s: str):
            """
            Returns the source for a C string literal or the given string, including
            the explicit null at the end
            """
            return '"{}\\0"'.format(json.dumps(s).strip('"'))

        parts: List[str] = []
        parts.append("// Array of characters, referenced by IndexRanges elsewhere.")
        parts.append("// Each IndexRange denotes a string.")
        parts.append('static const char kStrings[] =');
        parts.extend(['  {} // {}'.format(c_str(s), str(self.context.strings[s])) for s in self.context.string_buffer])
        parts.append(';\n');
        self.body_decls.extend(parts);

        parts: List[str] = []
        parts.append("""// Array of IndexRanges, where each represents a string by referencing
// the kStrings table.
// This array contains all sequences of alias strings used in the grammar.
// This table is referenced by an IndexRange elsewhere, i.e. by the 'aliases'
// field of an instruction or operand description.""")
        parts.append('static const IndexRange kAliasSpans[] = {');
        ranges = self.context.range_buffer['alias']
        for i in range(0, len(ranges)):
            ir = ranges[i]
            parts.append('  {}, // {} {}'.format(str(ir), i, self.context.GetString(ir)))
        parts.append('};\n');
        self.body_decls.extend(parts);

        parts = []
        parts.append("// Array of capabilities, referenced by IndexRanges elsewhere.")
        parts.append("// Contains all sequences of capabilities used in the grammar.")
        parts.append('static const spv::Capability kCapabilitySpans[] = {');
        capability_ranges = self.context.range_buffer['capability']
        for i in range(0, len(capability_ranges)):
            ir = capability_ranges[i]
            cap = self.context.GetString(ir)
            parts.append('  spv::Capability::{}, // {}'.format(cap, i))
        parts.append('};\n');
        self.body_decls.extend(parts);

        parts = []
        parts.append("// Array of extensions, referenced by IndexRanges elsewhere.")
        parts.append("// Contains all sequences of extensions used in the grammar.")
        parts.append('static const spvtools::Extension kExtensionSpans[] = {');
        ranges = self.context.range_buffer['extension']
        for i in range(0, len(ranges)):
            ir = ranges[i]
            name = self.context.GetString(ir)
            parts.append('  spvtools::Extension::k{}, // {}'.format(name, i))
        parts.append('};\n');
        self.body_decls.extend(parts);

        parts = []
        parts.append("// Array of operand types, referenced by IndexRanges elsewhere.")
        parts.append("// Contains all sequences of operand types used in the grammar.")
        parts.append('static const spv_operand_type_t kOperandSpans[] = {');
        ranges = self.context.range_buffer['operand']
        for i in range(0, len(ranges)):
            ir = ranges[i]
            name = self.context.GetString(ir)
            parts.append('  {}, // {}'.format(name, i))
        parts.append('};\n');
        self.body_decls.extend(parts);

        parts: List[str] = []
        parts.append("// Returns the name of an extension, as an index into kStrings")
        parts.append("IndexRange ExtensionToIndexRange(Extension extension) {\n  switch(extension) {")
        for e in self.extensions:
            parts.append('    case Extension::k{}: return {};'.format(e,self.context.AddString(e)))
        parts.append("    default: break;");
        parts.append('  }\n  return {};\n}\n');

        self.body_decls.extend(parts)


def make_path_to_file(f: str) -> None:
    """Makes all ancestor directories to the given file, if they don't yet
    exist.

    Arguments:
        f: The file whose ancestor directories are to be created.
    """
    dir = os.path.dirname(os.path.abspath(f))
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(dir):
            pass
        else:
            raise


def get_extension_list(instructions, operand_kinds):
    """Returns extensions as an alphabetically sorted list of strings.

    Args:
      instructions: list of instruction objects, using the JSON grammar file schema
      operand_kinds: list of operand_kind objects, using the JSON grammar file schema
    """

    things_with_an_extensions_field = [item for item in instructions]

    enumerants = sum([item.get('enumerants', [])
                      for item in operand_kinds], [])

    things_with_an_extensions_field.extend(enumerants)

    extensions = sum([item.get('extensions', [])
                      for item in things_with_an_extensions_field
                      if item.get('extensions')], [])

    for item in EXTENSIONS_FROM_SPIRV_REGISTRY_AND_NOT_FROM_GRAMMARS.split():
            # If it's already listed in a grammar, then don't put it in the
            # special exceptions list.
        assert item not in extensions, 'Extension %s is already in a grammar file' % item

    extensions.extend(
        EXTENSIONS_FROM_SPIRV_REGISTRY_AND_NOT_FROM_GRAMMARS.split())

    # Validator would ignore type declaration unique check. Should only be used
    # for legacy autogenerated test files containing multiple instances of the
    # same type declaration, if fixing the test by other methods is too
    # difficult. Shouldn't be used for any other reasons.
    extensions.append('SPV_VALIDATOR_ignore_type_decl_unique')

    return sorted(set(extensions))


def precondition_operand_kinds(operand_kinds):
    """For operand kinds that have the same number, make sure they all have the
    same extension list."""

    # Map operand kind and value to list of the union of extensions
    # for same-valued enumerants.
    exts = {}
    for kind_entry in operand_kinds:
        kind = kind_entry.get('kind')
        for enum_entry in kind_entry.get('enumerants', []):
            value = enum_entry.get('value')
            key = kind + '.' + str(value)
            if key in exts:
                exts[key].extend(enum_entry.get('extensions', []))
            else:
                exts[key] = enum_entry.get('extensions', [])
            exts[key] = sorted(set(exts[key]))

    # Now make each entry the same list.
    for kind_entry in operand_kinds:
        kind = kind_entry.get('kind')
        for enum_entry in kind_entry.get('enumerants', []):
            value = enum_entry.get('value')
            key = kind + '.' + str(value)
            if len(exts[key]) > 0:
                enum_entry['extensions'] = exts[key]

    return operand_kinds


def prefix_operand_kind_names(prefix, json_dict):
    """Modifies json_dict, by prefixing all the operand kind names
    with the given prefix.  Also modifies their uses in the instructions
    to match.
    """

    old_to_new = {}
    for operand_kind in json_dict["operand_kinds"]:
        old_name = operand_kind["kind"]
        new_name = prefix + old_name
        operand_kind["kind"] = new_name
        old_to_new[old_name] = new_name

    for instruction in json_dict["instructions"]:
        for operand in instruction.get("operands", []):
            replacement = old_to_new.get(operand["kind"])
            if replacement is not None:
                operand["kind"] = replacement


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate SPIR-V info tables')

    parser.add_argument('--spirv-core-grammar', metavar='<path>',
                        type=str, required=False,
                        help='input JSON grammar file for core SPIR-V '
                        'instructions')
    parser.add_argument('--extinst-debuginfo-grammar', metavar='<path>',
                        type=str, required=False, default=None,
                        help='input JSON grammar file for DebugInfo extended '
                        'instruction set')
    parser.add_argument('--extinst-cldebuginfo100-grammar', metavar='<path>',
                        type=str, required=False, default=None,
                        help='input JSON grammar file for OpenCL.DebugInfo.100 '
                        'extended instruction set')
    parser.add_argument('--extinst-glsl-grammar', metavar='<path>',
                        type=str, required=False, default=None,
                        help='input JSON grammar file for GLSL extended '
                        'instruction set')
    parser.add_argument('--extinst-opencl-grammar', metavar='<path>',
                        type=str, required=False, default=None,
                        help='input JSON grammar file for OpenCL extended '
                        'instruction set')

    parser.add_argument('--core-tables-body-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for core SPIR-V grammar tables to be included in .cpp')
    parser.add_argument('--core-tables-header-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for core SPIR-V grammar tables to be included in .h')

    # TODO: remove unused options
    parser.add_argument('--core-insts-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for core SPIR-V instructions')
    parser.add_argument('--glsl-insts-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for GLSL extended instruction set')
    parser.add_argument('--opencl-insts-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for OpenCL extended instruction set')
    parser.add_argument('--operand-kinds-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for operand kinds')
    parser.add_argument('--extension-enum-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for extension enumeration')
    parser.add_argument('--enum-string-mapping-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for enum-string mappings')
    parser.add_argument('--extinst-vendor-grammar', metavar='<path>',
                        type=str, required=False, default=None,
                        help='input JSON grammar file for vendor extended '
                        'instruction set'),
    parser.add_argument('--vendor-insts-output', metavar='<path>',
                        type=str, required=False, default=None,
                        help='output file for vendor extended instruction set')
    parser.add_argument('--vendor-operand-kind-prefix', metavar='<string>',
                        type=str, required=False, default=None,
                        help='prefix for operand kinds (to disambiguate operand type enums)')
    args = parser.parse_args()


    # The GN build system needs this because it doesn't handle quoting
    # empty string arguments well.
    if args.vendor_operand_kind_prefix == "...nil...":
        args.vendor_operand_kind_prefix = ""

    if (args.core_insts_output is None) != \
            (args.operand_kinds_output is None):
        print('error: --core-insts-output and --operand-kinds-output '
              'should be specified together.')
        exit(1)
    if args.operand_kinds_output and not (args.spirv_core_grammar and
         args.extinst_debuginfo_grammar and
         args.extinst_cldebuginfo100_grammar):
        print('error: --operand-kinds-output requires --spirv-core-grammar '
              'and --extinst-debuginfo-grammar '
              'and --extinst-cldebuginfo100-grammar')
        exit(1)
    if (args.glsl_insts_output is None) != \
            (args.extinst_glsl_grammar is None):
        print('error: --glsl-insts-output and --extinst-glsl-grammar '
              'should be specified together.')
        exit(1)
    if (args.opencl_insts_output is None) != \
            (args.extinst_opencl_grammar is None):
        print('error: --opencl-insts-output and --extinst-opencl-grammar '
              'should be specified together.')
        exit(1)
    if (args.vendor_insts_output is None) != \
            (args.extinst_vendor_grammar is None):
        print('error: --vendor-insts-output and '
              '--extinst-vendor-grammar should be specified together.')
        exit(1)
    if all([args.core_insts_output is None,
            args.core_tables_body_output is None,
            args.core_tables_header_output is None,
            args.glsl_insts_output is None,
            args.opencl_insts_output is None,
            args.vendor_insts_output is None,
            args.extension_enum_output is None,
            args.enum_string_mapping_output is None]):
        print('error: at least one output should be specified.')
        exit(1)

    if args.spirv_core_grammar is not None:
        # Populate instructions, extensions, operand_kinds list of json objects
        with open(args.spirv_core_grammar) as json_file:
            core_grammar = json.loads(json_file.read())
            with open(args.extinst_debuginfo_grammar) as debuginfo_json_file:
                debuginfo_grammar = json.loads(debuginfo_json_file.read())
                with open(args.extinst_cldebuginfo100_grammar) as cldebuginfo100_json_file:
                    cldebuginfo100_grammar = json.loads(cldebuginfo100_json_file.read())
                    prefix_operand_kind_names("CLDEBUG100_", cldebuginfo100_grammar)
                    instructions = []
                    instructions.extend(core_grammar['instructions'])
                    instructions.extend(debuginfo_grammar['instructions'])
                    instructions.extend(cldebuginfo100_grammar['instructions'])
                    operand_kinds = []
                    operand_kinds.extend(core_grammar['operand_kinds'])
                    operand_kinds.extend(debuginfo_grammar['operand_kinds'])
                    operand_kinds.extend(cldebuginfo100_grammar['operand_kinds'])

                    extensions = get_extension_list(instructions, operand_kinds)
                    operand_kinds = precondition_operand_kinds(operand_kinds)

                    printing_class: List[str] = [e['tag'] for e in core_grammar['instruction_printing_class']]

        g = Grammar(extensions, operand_kinds, printing_class)

        g.ComputeOperandTables()
        g.ComputeInstructionTables(core_grammar['instructions'])
        g.ComputeLeafTables()

        if args.core_tables_body_output is not None:
            make_path_to_file(args.core_tables_body_output)
            with open(args.core_tables_body_output, 'w') as f:
                f.write('\n'.join(g.body_decls))
        if args.core_tables_header_output is not None:
            make_path_to_file(args.core_tables_header_output)
            with open(args.core_tables_header_output, 'w') as f:
                f.write('\n'.join(g.header_decls))


if __name__ == '__main__':
  main()
