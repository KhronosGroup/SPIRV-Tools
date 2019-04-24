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

#include <cassert>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <functional>

#include "source/fuzz/fuzzer.h"
#include "source/fuzz/protobufs/spirvfuzz.pb.h"
#include "source/fuzz/replayer.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/opt/log.h"
#include "source/spirv_fuzzer_options.h"
#include "source/util/string_utils.h"
#include "tools/io.h"
#include "tools/util/cli_consumer.h"

#include "google/protobuf/util/json_util.h"

namespace {

// Status and actions to perform after parsing command-line arguments.
enum FuzzActions { FUZZ_CONTINUE, FUZZ_REPLAY, FUZZ_STOP };

struct FuzzStatus {
  FuzzActions action;
  int code;
};

void PrintUsage(const char* program) {
  // NOTE: Please maintain flags in lexicographical order.
  printf(
      R"(%s - Fuzzes an equivalent SPIR-V binary based on a given binary.

USAGE: %s [options] <input>

The SPIR-V binary is read from <input>.

NOTE: The fuzzer is a work in progress.

Options (in lexicographical order):

  -h, --help
               Print this help.
  --replay
               File from which to read a sequence of transformations to replay
               (instead of fuzzing)
  --seed
               Unsigned 32-bit integer seed to control random number
               generation.
  --version
               Display fuzzer version information.

)",
      program, program);
}

// Message consumer for this tool.  Used to emit diagnostics during
// initialization and setup. Note that |source| and |position| are irrelevant
// here because we are still not processing a SPIR-V input file.
void FuzzDiagnostic(spv_message_level_t level, const char* /*source*/,
                    const spv_position_t& /*position*/, const char* message) {
  if (level == SPV_MSG_ERROR) {
    fprintf(stderr, "error: ");
  }
  fprintf(stderr, "%s\n", message);
}

FuzzStatus ParseFlags(int argc, const char** argv, const char** in_file,
                      std::unique_ptr<char>* replay_transformations_file,
                      spvtools::FuzzerOptions* fuzzer_options) {
  uint32_t positional_arg_index = 0;

  for (int argi = 1; argi < argc; ++argi) {
    const char* cur_arg = argv[argi];
    if ('-' == cur_arg[0]) {
      if (0 == strcmp(cur_arg, "--version")) {
        spvtools::Logf(FuzzDiagnostic, SPV_MSG_INFO, nullptr, {}, "%s\n",
                       spvSoftwareVersionDetailsString());
        return {FUZZ_STOP, 0};
      } else if (0 == strcmp(cur_arg, "--help") || 0 == strcmp(cur_arg, "-h")) {
        PrintUsage(argv[0]);
        return {FUZZ_STOP, 0};
      } else if (0 == strncmp(cur_arg, "--replay=", sizeof("--replay=") - 1)) {
        const auto split_flag = spvtools::utils::SplitFlagArgs(cur_arg);
        *replay_transformations_file =
            std::unique_ptr<char>(new char[split_flag.second.size() + 1]);
        strcpy(replay_transformations_file->get(), split_flag.second.c_str());
      } else if (0 == strncmp(cur_arg, "--seed=", sizeof("--seed=") - 1)) {
        const auto split_flag = spvtools::utils::SplitFlagArgs(cur_arg);
        char* end = nullptr;
        errno = 0;
        const auto seed =
            static_cast<uint32_t>(strtol(split_flag.second.c_str(), &end, 10));
        assert(end != split_flag.second.c_str() && errno == 0);
        fuzzer_options->set_random_seed(seed);
      } else if ('\0' == cur_arg[1]) {
        // We do not support fuzzing from standard input.  We could support
        // this if there was a compelling use case.
        PrintUsage(argv[0]);
        return {FUZZ_STOP, 0};
      } else {
      }
    } else if (positional_arg_index == 0) {
      // Input file name
      assert(!*in_file);
      *in_file = cur_arg;
      positional_arg_index++;
    } else {
      spvtools::Error(FuzzDiagnostic, nullptr, {},
                      "Too many positional arguments specified");
      return {FUZZ_STOP, 1};
    }
  }

  if (!*in_file) {
    spvtools::Error(FuzzDiagnostic, nullptr, {}, "No input file specified");
    return {FUZZ_STOP, 1};
  }

  if (*replay_transformations_file) {
    return {FUZZ_REPLAY, 0};
  }

  return {FUZZ_CONTINUE, 0};
}

}  // namespace

const auto kDefaultEnvironment = SPV_ENV_UNIVERSAL_1_3;

int main(int argc, const char** argv) {
  const char* in_file = nullptr;
  std::unique_ptr<char> replay_transformations_file = nullptr;

  spv_target_env target_env = kDefaultEnvironment;
  spvtools::FuzzerOptions fuzzer_options;

  FuzzStatus status = ParseFlags(argc, argv, &in_file,
                                 &replay_transformations_file, &fuzzer_options);

  if (status.action == FUZZ_STOP) {
    return status.code;
  }

  std::vector<uint32_t> binary_in;
  if (!ReadFile<uint32_t>(in_file, "rb", &binary_in)) {
    return 1;
  }
  std::vector<uint32_t> binary_out;
  spvtools::fuzz::protobufs::TransformationSequence transformations_applied;

  if (status.action == FUZZ_REPLAY) {
    std::ifstream existing_transformations_file;
    existing_transformations_file.open(replay_transformations_file.get(),
                                       std::ios::in | std::ios::binary);
    spvtools::fuzz::protobufs::TransformationSequence
        existing_transformation_sequence;
    auto parse_success = existing_transformation_sequence.ParseFromIstream(
        &existing_transformations_file);
    existing_transformations_file.close();
    if (!parse_success) {
      return 1;
    }
    spvtools::fuzz::Replayer replayer(target_env);
    replayer.SetMessageConsumer(spvtools::utils::CLIMessageConsumer);
    auto replay_result_status =
        replayer.Run(binary_in, existing_transformation_sequence, &binary_out,
                     &transformations_applied);
    if (replay_result_status !=
        spvtools::fuzz::Replayer::ReplayerResultStatus::kComplete) {
      return 1;
    }
  } else {
    spvtools::fuzz::Fuzzer fuzzer(target_env);
    fuzzer.SetMessageConsumer(spvtools::utils::CLIMessageConsumer);
    auto fuzz_result_status = fuzzer.Run(
        binary_in, &binary_out, &transformations_applied, fuzzer_options);
    if (fuzz_result_status !=
        spvtools::fuzz::Fuzzer::FuzzerResultStatus::kComplete) {
      return 1;
    }
  }

  if (!WriteFile<uint32_t>("_spirvfuzzoutput.spv", "wb", binary_out.data(),
                           binary_out.size())) {
    return 1;
  }

  std::ofstream transformations_file;
  transformations_file.open("_spirvfuzzoutput.transformations",
                            std::ios::out | std::ios::binary);
  bool success =
      transformations_applied.SerializeToOstream(&transformations_file);
  transformations_file.close();
  if (!success) {
    return 2;
  }

  std::string json_string;
  auto json_options = google::protobuf::util::JsonOptions();
  json_options.add_whitespace = true;
  auto json_generation_status = google::protobuf::util::MessageToJsonString(
      transformations_applied, &json_string, json_options);
  if (json_generation_status != google::protobuf::util::Status::OK) {
    return 3;
  }

  std::ofstream transformations_json_file("_spirvfuzzoutput.json");
  transformations_json_file << json_string;
  transformations_json_file.close();
}
