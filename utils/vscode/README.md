# Visual Studio Code extension for SPIR-V disassembly files

This directory holds a Visual Studio Code language server extension for SPIR-V assembly files (`.spvasm`)

The extension supports:
* Syntax highlighting
* Jump to definition
* Find all references
* Symbol renaming
* Operand hover information
* Formatting
* Completion suggestions for all Opcodes and Ids

## Dependencies

In order to build and install the Visual Studio Code language server extension, you will need to install and have on your `PATH` the following dependencies:
* [`npm`](https://www.npmjs.com/)
* [`golang 1.16+`](https://golang.org/)

## Installing (macOS / Linux)

Run `install_vscode.sh`

## Installing (Windows)

Run `install_vscode.bat`

## Installing the LSP for use outside of VSCode (masOS / Linux)

Run `install_lsp.sh`
Copy `spirvls` and `spirv.json` to a location in `$PATH`
Configure your LSP client as appropiate

## Installing the LSP for use outside of VSCode (Windows)

Run `install_lsp.bat`
Copy `spirvls` and `spirv.json` to a location in ??
Configure your LSP client as appropiate
