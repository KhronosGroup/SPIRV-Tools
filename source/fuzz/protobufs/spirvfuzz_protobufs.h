// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_FUZZ_SPIRVFUZZ_PROTOBUFS_H_
#define SOURCE_FUZZ_SPIRVFUZZ_PROTOBUFS_H_

// This header file serves to act as a barrier between the protobuf header
// files and files that include them.  It uses compiler pragmas to disable
// diagnostics, in order to ignore warnings generated during the processing
// of these header files without having to compromise on freedom from warnings
// in the rest of the project.

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// The following should be the only place in the project where protobuf files
// are directly included.  This is so that they can be compiled in a manner
// where warnings are ignored.

#include "google/protobuf/util/json_util.h"
#include "source/fuzz/protobufs/spirvfuzz.pb.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#endif  // SOURCE_FUZZ_SPIRVFUZZ_PROTOBUFS_H_
