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

#ifndef LIBSPIRV_OPT_PASS_REGISTRY_IMPL_H
#define LIBSPIRV_OPT_PASS_REGISTRY_IMPL_H

#include "pass_registry.h"

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace spvtools {
namespace opt {

// PassRegtryImpl is the real implementation of pass registry that holds a
// table of registered passes with their command line argument, pass maker
// functions and pass deleter functions.
class PassRegistryImpl {
 protected:
   // TODO(qining): These are left in protected section for testing.
  PassRegistryImpl() : pass_create_info_() {}
  ~PassRegistryImpl() {};

  // The internal bookkeeping for all registered pass creating info.
  std::unordered_map<std::string, std::pair<MakePassPfn, DeletePassPfn> >
      pass_create_info_;

 public:
  // Registers a pass with its command line argument, pass maker function and
  // pass deleter function. Returns true if succeeded, othwise false.
  bool Register(const char* cmd_arg, MakePassPfn make_pass,
                DeletePassPfn delete_pass) {
    if (pass_create_info_.count(cmd_arg) != 0) {
      return false;
    }
    pass_create_info_.insert({cmd_arg, std::make_pair(make_pass, delete_pass)});
    return true;
  }

  // Lists the command line arguments of all the registered passes.
  void List() {
    std::cout << "Command arguments of registered passes:" << std::endl;
    for (auto& p : pass_create_info_) {
      std::cout << p.first << std::endl;
    }
  }

  // Creates a pass instance with the given command line argument. Returns an
  // unique pointer built with the pass deleter function if a pass can be found
  // with the given command line argument, otherwise, returns a nullptr
  // (wrapped as an unique_ptr).
  std::unique_ptr<Pass, DeletePassPfn> GetPass(const char* cmd_arg) {
    if (pass_create_info_.count(cmd_arg) == 0) {
      return std::unique_ptr<Pass, DeletePassPfn>(nullptr, [](Pass*){});
    }
    auto& create_info = pass_create_info_[cmd_arg];
    return std::unique_ptr<Pass, DeletePassPfn>(create_info.first(),
                                                create_info.second);
  }

  // Get the pass registry impl instance, which is supposed to be a singleton.
  static PassRegistryImpl* GetPassRegistryImpl();
};
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_PASS_REGISTRY_IMPL_H
