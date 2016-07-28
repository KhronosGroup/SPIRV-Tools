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

// This is the public pass registry header which should be included in all
// loadable pass module through pass.h.

#ifndef LIBSPIRV_OPT_PASSREGISTRY_H
#define LIBSPIRV_OPT_PASSREGISTRY_H

namespace spvtools {
namespace opt {

class Pass;
class PassRegistryImpl;

// Define the pass maker and deleter function signatures. Confine the calling
// convention to be C Calling Convention.
using MakePassPfn = Pass* /* __attribute((cdecl)) */(*)();
using DeletePassPfn = void /* __attribute((cdecl)) */(*)(Pass*);

// A pass maker function, which wraps the default constructor of a pass type.
template <typename PassT>
// __attribute((cdecl))
static Pass* MakePass() {
  return new PassT;
}

// A pass deleter function, which wraps the default destructor of a pass type.
template <typename PassT>
// __attribute((cdecl))
static void DeletePass(Pass* pass) {
  delete static_cast<PassT*>(pass);
}

// PassRegistry holds all the registered passes. A pass should be registered
// with a command line argument as an identifier, pass maker function and pass
// deleter function.
class PassRegistry {
 public:
  // Returns true if a pass is registered successfully, otherwise return false.
  bool Register(const char* cmd_arg, MakePassPfn make_pass,
                DeletePassPfn delete_pass);

  static PassRegistry* GetPassRegistry();

 private:
  PassRegistry() = delete;
  ~PassRegistry() {};
  PassRegistry(PassRegistryImpl* impl) : impl_(impl) {}
  PassRegistryImpl* impl_;
};


// RegisterPass registers a pass type to the registry.
template <typename PassT>
struct RegisterPass {
  RegisterPass(const char* cmd_arg) {
    PassRegistry::GetPassRegistry()->Register(cmd_arg, &MakePass<PassT>,
                                              &DeletePass<PassT>);
  }
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_PASSREGISTRY_H
