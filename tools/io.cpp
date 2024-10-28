// Copyright (c) 2024 Google Inc.
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

#include "io.h"

#include <assert.h>

#if defined(SPIRV_WINDOWS)
#include <fcntl.h>
#include <io.h>

#define SET_STDIN_TO_BINARY_MODE() _setmode(_fileno(stdin), O_BINARY);
#define SET_STDIN_TO_TEXT_MODE() _setmode(_fileno(stdin), O_TEXT);
#define SET_STDOUT_TO_BINARY_MODE() _setmode(_fileno(stdout), O_BINARY);
#define SET_STDOUT_TO_TEXT_MODE() _setmode(_fileno(stdout), O_TEXT);
#define SET_STDOUT_MODE(mode) _setmode(_fileno(stdout), mode);
#else
#define SET_STDIN_TO_BINARY_MODE()
#define SET_STDIN_TO_TEXT_MODE()
#define SET_STDOUT_TO_BINARY_MODE() 0
#define SET_STDOUT_TO_TEXT_MODE() 0
#define SET_STDOUT_MODE(mode)
#endif

namespace {
// Appends the contents of the |file| to |data|, assuming each element in the
// file is of type |T|.
template <typename T>
void ReadFile(FILE* file, std::vector<T>* data) {
  if (file == nullptr) return;

  const int buf_size = 1024;
  T buf[buf_size];
  while (size_t len = fread(buf, sizeof(T), buf_size, file)) {
    data->insert(data->end(), buf, buf + len);
  }
}

// Returns true if |file| has encountered an error opening the file or reading
// the file as a series of element of type |T|. If there was an error, writes an
// error message to standard error.
template <class T>
bool WasFileCorrectlyRead(FILE* file, const char* filename) {
  if (file == nullptr) {
    fprintf(stderr, "error: file does not exist '%s'\n", filename);
    return false;
  }

  if (ftell(file) == -1L) {
    if (ferror(file)) {
      fprintf(stderr, "error: error reading file '%s'\n", filename);
      return false;
    }
  } else {
    if (sizeof(T) != 1 && (ftell(file) % sizeof(T))) {
      fprintf(
          stderr,
          "error: file size should be a multiple of %zd; file '%s' corrupt\n",
          sizeof(T), filename);
      return false;
    }
  }
  return true;
}
}  // namespace

bool ReadBinaryFile(const char* filename, std::vector<uint32_t>* data) {
  assert(data->empty());

  const bool use_file = filename && strcmp("-", filename);
  FILE* fp = nullptr;
  if (use_file) {
    fp = fopen(filename, "rb");
  } else {
    SET_STDIN_TO_BINARY_MODE();
    fp = stdin;
  }

  ReadFile(fp, data);
  bool succeeded = WasFileCorrectlyRead<uint32_t>(fp, filename);
  if (use_file && fp) fclose(fp);
  return succeeded;
}

bool ReadTextFile(const char* filename, std::vector<char>* data) {
  assert(data->empty());

  const bool use_file = filename && strcmp("-", filename);
  FILE* fp = nullptr;
  if (use_file) {
    fp = fopen(filename, "r");
  } else {
    SET_STDIN_TO_TEXT_MODE();
    fp = stdin;
  }

  ReadFile(fp, data);
  bool succeeded = WasFileCorrectlyRead<char>(fp, filename);
  if (use_file && fp) fclose(fp);
  return succeeded;
}

namespace {
// A class to create and manage a file for outputting data.
class OutputFile {
 public:
  // Opens |filename| in the given mode.  If |filename| is nullptr, the empty
  // string or "-", stdout will be set to the given mode.
  OutputFile(const char* filename, const char* mode) : old_mode_(0) {
    const bool use_stdout =
        !filename || (filename[0] == '-' && filename[1] == '\0');
    if (use_stdout) {
      if (strchr(mode, 'b')) {
        old_mode_ = SET_STDOUT_TO_BINARY_MODE();
      } else {
        old_mode_ = SET_STDOUT_TO_TEXT_MODE();
      }
      fp_ = stdout;
    } else {
      fp_ = fopen(filename, mode);
    }
  }

  ~OutputFile() {
    if (fp_ == stdout) {
      fflush(stdout);
      SET_STDOUT_MODE(old_mode_);
    } else if (fp_ != nullptr) {
      fclose(fp_);
    }
  }

  // Returns a file handle to the file.
  FILE* GetFileHandle() const { return fp_; }

 private:
  FILE* fp_;
  int old_mode_;
};
}  // namespace

template <typename T>
bool WriteFile(const char* filename, const char* mode, const T* data,
               size_t count) {
  OutputFile file(filename, mode);
  FILE* fp = file.GetFileHandle();
  if (fp == nullptr) {
    fprintf(stderr, "error: could not open file '%s'\n", filename);
    return false;
  }

  size_t written = fwrite(data, sizeof(T), count, fp);
  if (count != written) {
    fprintf(stderr, "error: could not write to file '%s'\n", filename);
    return false;
  }

  return true;
}

template bool WriteFile<uint32_t>(const char* filename, const char* mode,
                                  const uint32_t* data, size_t count);
template bool WriteFile<char>(const char* filename, const char* mode,
                              const char* data, size_t count);
