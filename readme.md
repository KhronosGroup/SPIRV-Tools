# SPIR-V Tools

## Overview

The project includes an assembler, disassembler, and validator for SPIR-V, all
based on a common static library. The library contains all of the implementation
details and is used in the standalone tools whilst also enabling integration
into other code bases directly.

Currently, the assembler and disassembler only support the core SPIR-V
specification (i.e. nothing Vulkan or OpenCL-specific) and the validator is a
work in progress. See the Future Work section for more information.

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

## Tools

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

#### Format

The assembly attempts to adhere to the binary form as closely as possible using
text names from that specification. Here is an example.

```
OpCapability Shader
OpMemoryModel Logical Simple
OpEntryPoint GLCompute $3 "main"
OpExecutionMode $3 LocalSize 64 64 1
OpTypeVoid %1
OpTypeFunction %2 $1
OpFunction $1 %3 None $2
OpLabel %4
OpReturn
OpFunctionEnd
```

Each line encapsulates one and only one instruction, or an OpCode and all of its
operands. OpCodes use the names provided in section 3.28 Instructions of the
SPIR-V specification, immediate values such as Addressing Model, Memory Model,
etc. use the names provided in sections 3.2 Source Language through 3.27
Capability of the SPIR-V specification. Literals strings are enclosed in quotes
`"<string>"` while literal numbers have no special formatting.

##### ID Definitions & Usage

An ID definition pertains to the `Result <id>` of an OpCode, and ID usage is any
input to an OpCode. To differentiate between definitions and uses, all ID
definitions are prefixed with `%` and take the form `%<id>`, meanwhile all ID
uses are prefixed with `$` and take the form `$<id>`. See the above example to
see this in action.

##### Named IDs

The assembler also supports named IDs, or virtual IDs, which greatly improves
the readability of the assembly. The same ID definition and usage prefixes
apply. Names must begin with an character in the range `[a-z|A-Z]`. The
following example will result in identical SPIR-V binary as the example above.

```
OpCapability Shader
OpMemoryModel Logical Simple
OpEntryPoint GLCompute $main "main"
OpExecutionMode $main LocalSize 64 64 1
OpTypeVoid %void
OpTypeFunction %fnMain $void
OpFunction $void %main None $fnMain
OpLabel %lbMain
OpReturn
OpFunctionEnd
```

##### Arbitrary Integers

When writing tests it can be useful to emit an invalid 32 bit word into the
binary stream at arbitrary positions within the assembly. To specify an
arbitrary word into the stream the prefix `!` is used, this takes the form
`!<integer>`. Here is an example.

```
OpCapability !0x0000FF000
```

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
use the [googletest](https://code.google.com/p/googletest/) framework. The
[googletest](https://code.google.com/p/googletest/) source is not provided with
this project, to enable the tests place the
[googletest](https://code.google.com/p/googletest/) source in the
`<spirv-dir>/external/googletest` directory, rerun CMake if you have already
done so previously, CMake will detect the existence of
`<spirv-dir>/external/googletest` then build as normal.

## Future Work

* Support extension libraries in `spirv-as`, `spirv-dis`, and `spirv-val`.
* Complete implementation of ID validation rules in `spirv-val`.
* Implement section 2.16 Validation Rules in `spirv-val`.
* Implement Capability validation and or report in `spirv-val`.
* Improve assembly output from `spirv-dis`.
* Improve diagnostic reports.

## Known Issues

* Improve literal parsing in the assembler, currently only decimal integers and
  floating-point numbers are supported as literal operands and the parser is not
  contextually aware of the desired width of the operand.

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
