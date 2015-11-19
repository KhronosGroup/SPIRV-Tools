# SPIR-V Tools

## Overview

The SPIR-V Tools project provides an API and commands for processing SPIR-V
modules.

The project includes an assembler, binary module parser, disassembler, and
validator for SPIR-V, all based on a common static library. The library contains
all of the implementation details, and is used in the standalone tools whilst
also enabling integration into other code bases directly.

The interfaces are still under development, and are expected to change.

SPIR-V is defined by the Khronos Group Inc.
See the [SPIR-V Registry](https://www.khronos.org/registry/spir-v/) for the
SPIR-V specification, headers, and XML registry.

## Supported features

### Assembler, binary parser, and disassembler

* Based on SPIR-V 1.0 Revision 2.
  * Supports GLSL std450 extended instructions.
  * Supports OpenCL extended instructions.
* Assembler only does basic syntax checking.  No cross validation of
  IDs or types is performed, except to check literal arguments to
  `OpConstant`, `OpSpecConstant`, and `OpSwitch`.

See [`syntax.md`](syntax.md) for the assembly language syntax.

### Validator

*Warning:* The validator is incomplete.

## Source code

The SPIR-V Tools are maintained by members of the The Khronos Group Inc.,
at [https://github.com/KhronosGroup/SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools).

Contributions via merge request are welcome. Changes should:
* Be provided under the [Khronos license](#license).
* Include tests to cover updated functionality.
* C++ code should follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
* Code should be formatted with `clang-format`.  Settings are defined by
  the included [.clang-format](.clang-format) file.

We intend to maintain a linear history on the GitHub `master` branch.

### Source code organization

* `external/headers`: Standard SPIR-V header files used by the implementation,
   from the SPIR-V Registry
* `external/googletest`: Intended location for the
  [googletest](https://github.com/google/googletest) sources, not provided.
* `include/libspirv/libspirv.h`: C API public interface
* `source/`: API implementation
* `test/`: Tests, using the [googletest](https://github.com/google/googletest)
  framework.
* `tools/`: Command line executables

### Tests

The project contains a number of tests, used to drive development
and ensure correctness.  The tests are written using the
[googletest](https://github.com/google/googletest) framework.  The `googletest`
source is not provided with this project.  Download the `googletest` source
into the `<spirv-dir>/external/googletest` directory before configuring
and building the project.

## Build

The project uses [CMake](https://cmake.org/) to generate platform-specific
build configurations.  To generate these build files, issue the following
commands:

```
mkdir <spirv-dir>/build
cd <spirv-dir>/build
cmake [-G <platform-generator>] <spirv-dir>
```

Once the build files have been generated, build using your preferred
development environment.

### CMake options

The following CMake options are supported:

* `SPIRV_COLOR_TERMINAL=ON` - Enables color console output, enabled by default.
* `SPIRV_SKIP_EXECUTABLES=ON` - Build only the library, not the command line
  tools.  This will also prevent the tests from being built.
* `SPIRV_USE_SANITIZER=<sanitizer>` - On UNIX platforms with an appropriate
  version of `clang` this option enables the use of the sanitizers documented
  [here](http://clang.llvm.org/docs/UsersManual.html#controlling-code-generation).
  This should only be used with a debug build. Disabled by default.
* `SPIRV_WARN_EVERYTHING=OFF` - On UNIX platforms enable the `-Weverything`
  compiler front end option, disabled by default.
* `SPIRV_WERROR=ON` - Forces a compilation error on any warnings encountered by
  enabling the compiler-specific compiler front-end option, enabled by default.

## Library

### Usage

The library provides a C API, but the internals use C++11.

In order to use the library from an application, the include path should point to
`<spirv-dir>/include`, which will enable the application to include the header
`<spirv-dir>/include/libspirv/libspirv.h` then linking against the static
library in `<spirv-build-dir>/bin/libSPIRV-Tools.a` or
`<spirv-build-dir>/bin/SPIRV-Tools.lib`.

* `SPIRV-Tools` CMake target: Creates the static library:
  * `<spirv-build-dir>/lib/libSPIRV-Tools.a` on Linux and OS X.
  * `<spirv-build-dir>/lib/libSPIRV-Tools.lib` on Windows.

#### Entry points

The interfaces are still under development, and are expected to change.

There are three main entry points into the library.

* `spvTextToBinary`: An assembler, translating text to a binary SPIR-V module.
* `spvBinaryToText`: A disassembler, translating a binary SPIR-V module to
  text.
* `spvBinaryParse`: The entry point to a binary parser API.  It issues callbacks
  for the header and each parsed instruction.  The disassembler is implemented
  as a client of `spvBinaryParse`.
* `spvValidate` implements the validator functionality. *Incomplete*

## Command line tools

### Assembler tool

The assembler reads the assembly language text, and emits the binary form.

The standalone assembler is the exectuable called `spirv-as`, and is located in
`<spirv-build-dir>/bin/spirv-as`.  The functionality of the assembler is
implemented by the `spvTextToBinary` library function.

* `spirv-as` - the standalone assembler
  * `<spirv-dir>/bin/spirv-as`

Use option `-h` to print help.

### Disassembler tool

The disassembler reads the binary form, and emits assembly language text.

The standalone disassembler is the executable called `spirv-dis`, and is located in
`<spirv-build-dir>/bin/spirv-dis`. The functionality of the disassembler is
implemented by the `spvBinaryToText` library function.

* `spirv-dis` - the standalone disassembler
  * `<spirv-dir>/bin/spirv-dis`

Use option `-h` to print help.

The output includes syntax colouring when printing to the standard output stream,
on Linux, Windows, and OS X.

### Validator tool

*Warning:* This functionality is under development, and is incomplete.

The standalone validator is the executable called `spirv-val`, and is located in
`<spirv-build-dir>/bin/spirv-val`. The functionality of the validator is
implemented by the `spvValidate` library function.

The validator operates on the binary form.

* `spirv-val` - the standalone validator
  * `<spirv-dir>/bin/spirv-val`

### UnitSPIRV tool

The `<spirv-build-dir>/bin/UnitSPIRV` executable runs the project tests.
It supports the standard `googletest` command line options.

## Future Work
<a name="future"></a>

### Assembler and disassembler

* Support 16-bit floating point literals.
* The disassembler could emit helpful annotations in comments.  For example:
  * Use variable name information from debug instructions to annotate
    key operations on variables.
  * Show control flow information by annotating `OpLabel` instructions with
    that basic block's predecessors.
* Error messages could be improved.

### Validator

This is a work in progress.

## Licence
<a name="license"></a>
```
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
```
