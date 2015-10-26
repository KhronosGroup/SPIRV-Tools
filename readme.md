# SPIR-V Tools

## Overview

The project includes an assembler, disassembler, and validator for SPIR-V, all
based on a common static library. The library contains all of the implementation
details and is used in the standalone tools whilst also enabling integration
into other code bases directly.

## Supported features

### Assembler and disassembler

* Based on SPIR-V 0.99 Revision 32
  * Supports core instructions and enumerants from Rev 32.
* All GLSL std450 extended instructions are supported.
* Assembler only does basic syntax checking.  No cross validation of
  IDs or types is performed, except to check literal arguments to
  `OpConstant`, `OpSpecConstant`, and `OpSwitch`.
* OpenCL extended instructions are supported, from version 1.0 Revision 1.

### Validator

The validator is incomplete.  See the Future Work section for more information.

## CHANGES (for tools hackers)

2015-10-16
* OpenCL extended instructions are supported, from version 1.0 Revision 1.
* Capability dependencies for instructions and enums now match 0.99 Rev 32.
* Very long instructions are supported, up to SPIR-V universal limits.
* UTF-8 literal strings are supported.
* Assembler support for numeric literals:
   * 32 and 64-bit floating point literals are encoded properly.
   * Signed and unsigned integers of any width up to 64 bits are supported
     and encoded properly.
   * Hexadecimal literals are supported.  See `syntax.md`.
   * Numeric literal arguments to `OpConstant`, `OpSpecConstant`, and `OpSwitch`
     are type- and range-checked.
     The assembler checks that other literal numbers are non-negative integers.
     That's the grammar works, at least for now.

2015-10-02
* Completed assembler support for [`!<integer>` syntax](syntax.md#immediate)
  to inject arbitrary words into the binary. This feature is used
  for testing.
* Internal cleanups and additional tests.

2015-09-25
* Updated to Rev32 headers
  * Core instructions and enumerants from Rev 32 are supported by the
    assembler.
  * Capability dependencies may be incomplete or incorrect.
* Assembler ID syntax:
  * All IDs must use the `%` prefix before the name.
  * ID name syntax is checked: it must be made from letters, numbers or
    underscore (`_`).
  * IDs with different names always map to different ID numbers.
    Previously, a numeric ID such as `%2` could accidentally map to the
    same numeric ID as named ID `%foo`.
  * In particular, an ID with a number for a name doesn't necessarily
    map to that number. E.g. `%2` doesn't necessarily map to ID 2.
* Disassembler emits mask expressions for mask combinations instead of
  erroring out.
* Fixed parsing and printing of Execution Scope and Memory Semantics
  operands.  They are supplied as IDs, not as literals.

2015-09-18
* MILESTONE: This version of the assembler supports all of SPIR-V Rev31,
  provided you only use 32-bit values.
* Fixes build problems with MSVC 2013.
* Assembler supports mask expressions
  * e.g. OpStore %ptr %value Volatile|Aligned 4
  * See [`syntax.md`](syntax.md) for more.
* Assembler supports image operands from Rev31.
  * This uses mask expression support.
* Assembler supports enum operands:
  storage class enums, sampler addressing mode,
  sampler filter mode, dim, image format
* More support for `!<number>` syntax.  Still incomplete.
* Disassembler will print 64-bit values correctly.

2015-09-15
* Fixed spelling of Function Control "Inline" enumerated value.
* Fixed: `Aligned` memory access flag takes a literal number operand.
* Fixed parsing of scope ID arguments, e.g. for group operations.

2015-09-11
* Assembly format must be consistent across the entire source module.
  * Add API assembler and disassembler entry points to control the format.
  * Add command line options to assembler and disassembler to force it:
    --syntax-format=assignment
    --syntax-format=canonical
    The default is "assignment".
* Fixes decorations:
  * Names: SaturatedConversion, FuncParamAttr NoCapture
  * Values: Fixes values for some decorations: BuiltIn LocalInvocationId, and BuiltIn
    SubgroupId
  * All handling of FPFastMathMode masks.
  * LinkageAttributes now requires the literal string operand.
* Fixes capabilities: Adds ImageMipmap, and capabilities from LiteralSampler through
  SampleRateShading.

2015-09-09
* Avoid confusion about ownership of storage:
  * `spv_binary` is only used for output of the assembler, and should
  always be destroyed with `spvBinaryDestroy`.
  * `spv_text` is only used for output of the disassembler, and should
  always be destroyed with `spvTextDestroy`.
  * Inputs to the assembler and disassembler are provided as pointer
  and length arguments.
* Fixed parsing of floating point literals.
* Fixed the -p option for the disassembler executable.
* Fixed a build break on MSVC when using a ternary operator with conflicting
  types.
* More test coverage and other cleanups.

2015-09-04
* The parser has been overhauled
  * We use an automatically generated table to describe the syntax of each
    core instruction.  The changes to the SPIR-V spec document generator to
    create this table are still being developed.
  * The parser uses a dynamically updated list of expected operand types.
    It is expanded as needed for variable-length lists of operands, and
    consumed during the parse.  See the uses of `spv_operand_pattern_t`.
  * The syntax of enum operands and their potential operands is still
    hand-coded.  (That might change depending on the cost-benefit tradeoff.)
* We are actively increasing test coverage.
* We have tweaked the CMake build rules to make it easier to integrate
  into other packages.  Google is integrating SPIR-V Tools into the Vulkan
  conformance test suite.
* New code tends to use Google C++ style, including formatting as generated
  by `clang-format --style=google`.
* The spvBinaryToText and spvTextToBinary interfaces have been updated to
  remove a conceptual ambiguity that arises when cleaning up `spv_binary_t`
  and `spv_text_t` objects.


## Where is the code?

The `master` branch of the repository is maintained by
Kenneth Benzie `k.benzie@codeplay.com`.

Please submit any merge requests as stated in these
[instructions](https://cvs.khronos.org/wiki/index.php/How_to_access_and_use_the_Khronos_Gitlab_Repository).


## Build

The project uses CMake to generate platform-specific build configurations. To
generate these build files issue the following commands.

```
mkdir <spirv-dir>/build
cd <spirv-dir>/build
cmake [-G<platform-generator>] ..
```

Once the build files have been generated, build using your preferred
development environment.

### CMake Options

* `SPIRV_USE_SANITIZER=<sanitizer>` - on UNIX platforms with an appropriate
  version of `clang` this option enables the use of the sanitizers documented
  [here](http://clang.llvm.org/docs/UsersManual.html#controlling-code-generation),
  this should only be used with a debug build, disabled by default
* `SPIRV_COLOR_TERMINAL=ON` - enables color console output, enabled by default
* `SPIRV_WARN_EVERYTHING=OFF` - on UNIX platforms enable the `-Weverything`
  compiler front end option, disabled by default
* `SPIRV_WERROR=OFF` - on UNIX platforms enable the `-Werror` compiler front end
  option, disabled by default

## Library

### Usage

In order to use the library from an application, the include path should point to
`<spirv-dir>/include`, which will enable the application to include the header
`<spirv-dir>/include/libspirv/libspirv.h` then linking against the static
library in `<spirv-build-dir>/bin/libSPIRV.a` or
`<spirv-build-dir>/bin/SPIRV.lib`. The intention is for this to be a C API,
however currently it relies on the generated header `spirv.h` meaning this is
currently a C++ API.

* `SPIRV` - the static library CMake target outputs `<spirv-dir>/lib/libSPIRV.a`
  on Linux/Mac or `<spirv-dir>/lib/SPIRV.lib` on Windows.

#### Entry Points

There are three main entry points into the library.

* `spvTextToBinary` implements the assembler functionality.
* `spvBinaryToText` implements the disassembler functionality.
* `spvValidate` implements the validator functionality.

### Source

In addition to the interface header `<spirv-dir>/include/libspirv/libspirv.h`
the implementation source files reside in `<spirv-dir>/source/*`.

The parsers for the assembler and disassembler use a table describing the
syntax of each core instruction.  This table can be generated from the SPIR-V
document generator:

1. Apply the patch in `source/core_syntax_table.patch` to the document generator.
2. Run the document generator with the `-a` option and place the results in
the `opcode.inc` file in the SPIR-V Tools `source` directory.
3. Be aware of version skew: The SPIR-V document generator might target a newer
verison of the spec than targeted by the SPIR-V tools.

### Assembler

The standalone assembler is the binary called `spirv-as` and is located in
`<spirv-build-dir>/bin/spirv-as`. The functionality of the assembler is
implemented by the `spvTextToBinary` library function.

The assembler operates on the textual form.

* `spirv-as` - the standalone assembler
  * `<spirv-dir>/bin/spirv-as`

#### Options

* `-o <filename>` is used to specify the output file, otherwise this is set to
  `out.spv`.

#### Syntax

See [`syntax.md`](syntax.md) for the assembly language syntax.

### Disassembler

The standalone disassembler is the binary called `spirv-dis` and is located in
`<spirv-build-dir>/bin/spirv-dis`. The functionality of the disassembler is
implemented by the `spvBinaryToText` library function.

The disassembler operates on the binary form.

* `spirv-dis` - the standalone disassembler
  * `<spirv-dir>/bin/spirv-dis`

#### Options

* `-o <filename>` is used to specify the output file, otherwise this is set to
  `out.spvasm`.
* `-p` prints the assembly to the console on stdout, this includes colored
  output on Linux, Windows, and Mac.

### Validator

The standalone validator is the binary called `spirv-val` and is located in
`<spirv-build-dir>/bin/spirv-val`. The functionality of the validator is
implemented by the `spvValidate` library function.

The validator operates on the binary form.

* `spirv-val` - the standalone validator
  * `<spirv-dir>/bin/spirv-val`

#### Options

* `-basic` performs basic stream validation, currently not implemented.
* `-layout` performs logical layout validation as described in section 2.16
  Validation Rules, currently not implemented.
* `-id` performs ID validation according to the instruction rules in sections
  3.28.1 through 3.28.22, enabled but is a work in progress.
* `-capability` performs capability validation and or reporting, currently not
  implemented.

## Tests

The project contains a number of tests, implemented in the `UnitSPIRV`
executable, used to drive the development and correctness of the tools, these
use the [googletest](https://github.com/google/googletest) framework. The
[googletest](https://github.com/google/googletest) source is not provided with
this project, to enable the tests place the
[googletest](https://github.com/google/googletest) source in the
`<spirv-dir>/external/googletest` directory, rerun CMake if you have already
done so previously, CMake will detect the existence of
`<spirv-dir>/external/googletest` then build as normal.

## Future Work

### Assembler and disassembler

* WIP: Fix disassembler support for non-32-bit numeric literals
* WIP: Fix bug: Assembler can't use extended instructions from
  two different extended instruction imports.
* Support 16-bit floating point literals.

### Validator

* Adopt the parser strategy used by the text and binary parsers.
* Complete implementation of ID validation rules in `spirv-val`.
* Implement section 2.16 Validation Rules in `spirv-val`.
* Implement Capability validation and or report in `spirv-val`.
* Improve assembly output from `spirv-dis`.
* Improve diagnostic reports.

## Known Issues

* Header file `libspirv.h` cannot be used in C code.
* Improve literal parsing in the assembler, currently only decimal integers and
  floating-point numbers are supported as literal operands and the parser is not
  contextually aware of the desired width of the operand.
* Sometimes the assembler will succeed, but the disassembler will fail to
  disassemble the result. (Is this still true?)

## Licence

Copyright (c) 2015 The Khronos Group Inc.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and/or associated documentation files (the
"Materials"), to deal in the Materials without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Materials, and to
permit persons to whom the Materials are furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Materials.

MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
   https://www.khronos.org/registry/

THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
