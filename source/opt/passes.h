// Copyright (c) 2016 Google Inc.
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

#ifndef LIBSPIRV_OPT_PASSES_H_
#define LIBSPIRV_OPT_PASSES_H_

// A single header to include all passes.

#include "eliminate_dead_constant_pass.h"
#include "fold_spec_constant_op_and_composite_pass.h"
#include "inline_pass.h"
#include "freeze_spec_constant_value_pass.h"
#include "null_pass.h"
#include "set_spec_constant_default_value_pass.h"
#include "strip_debug_info_pass.h"
#include "unify_const_pass.h"

#endif  // LIBSPIRV_OPT_PASSES_H_
