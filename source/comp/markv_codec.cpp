// Copyright (c) 2017 Google Inc.
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

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "binary.h"
#include "diagnostic.h"
#include "enum_string_mapping.h"
#include "extensions.h"
#include "instruction.h"
#include "opcode.h"
#include "operand.h"
#include "spirv-tools/libspirv.h"
#include "spirv-tools/markv.h"
#include "spirv_endian.h"
#include "spirv_validator_options.h"
#include "util/bit_stream.h"
#include "util/parse_number.h"
#include "validate.h"
#include "val/instruction.h"
#include "val/validation_state.h"

using libspirv::Instruction;
using libspirv::ValidationState_t;
using spvtools::ValidateInstructionAndUpdateValidationState;
using spvutils::BitReaderWord64;
using spvutils::BitWriterWord64;

namespace {

const uint32_t kSpirvMagicNumber = 0x07230203;
const uint32_t kMarkvMagicNumber = 0x07230303;

const size_t kCommentNumWhitespaces = 2;

// TODO: This is a placeholder for a protobuf containing MARK-V model for a
// specific dataset.
class MarkvModel {
 public:
  size_t opcode_chunk_length() const { return 8; }
  size_t num_operands_chunk_length() const { return 3; }

  size_t u16_chunk_length() const { return 4; }
  size_t s16_chunk_length() const { return 4; }
  size_t s16_block_exponent() const { return 6; }

  size_t u32_chunk_length() const { return 8; }
  size_t s32_chunk_length() const { return 8; }
  size_t s32_block_exponent() const { return 10; }

  size_t u64_chunk_length() const { return 8; }
  size_t s64_chunk_length() const { return 8; }
  size_t s64_block_exponent() const { return 10; }
};

const MarkvModel* GetDefaultModel() {
  static MarkvModel model;
  return &model;
}

// Returns chunk length used for variable length encoding of spirv operand
// words. Returns zero if operand type corresponds to potentially multiple
// words or a word which is not expected to profit from variable width encoding.
// Chunk length is selected based on the size of expected value.
// Most ot these values will later be encoded with probability-based coding,
// but variable width integer coding is a good quick solution.
// TODO: Put this in MarkvModel proto.
size_t GetOperandVariableWidthChunkLength(spv_operand_type_t type) {
  switch (type) {
    case SPV_OPERAND_TYPE_TYPE_ID:
      return 4;
    case SPV_OPERAND_TYPE_RESULT_ID:
    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_SCOPE_ID:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
      return 8;
    case SPV_OPERAND_TYPE_LITERAL_INTEGER:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER:
      return 6;
    case SPV_OPERAND_TYPE_CAPABILITY:
      return 6;
    case SPV_OPERAND_TYPE_SOURCE_LANGUAGE:
    case SPV_OPERAND_TYPE_EXECUTION_MODEL:
      return 3;
    case SPV_OPERAND_TYPE_ADDRESSING_MODEL:
    case SPV_OPERAND_TYPE_MEMORY_MODEL:
      return 2;
    case SPV_OPERAND_TYPE_EXECUTION_MODE:
      return 6;
    case SPV_OPERAND_TYPE_STORAGE_CLASS:
      return 4;
    case SPV_OPERAND_TYPE_DIMENSIONALITY:
    case SPV_OPERAND_TYPE_SAMPLER_ADDRESSING_MODE:
      return 3;
    case SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE:
      return 2;
    case SPV_OPERAND_TYPE_SAMPLER_IMAGE_FORMAT:
      return 6;
    case SPV_OPERAND_TYPE_FP_ROUNDING_MODE:
    case SPV_OPERAND_TYPE_LINKAGE_TYPE:
    case SPV_OPERAND_TYPE_ACCESS_QUALIFIER:
    case SPV_OPERAND_TYPE_OPTIONAL_ACCESS_QUALIFIER:
      return 2;
    case SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE:
      return 3;
    case SPV_OPERAND_TYPE_DECORATION:
    case SPV_OPERAND_TYPE_BUILT_IN:
      return 6;
    case SPV_OPERAND_TYPE_GROUP_OPERATION:
    case SPV_OPERAND_TYPE_KERNEL_ENQ_FLAGS:
    case SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO:
      return 2;
    case SPV_OPERAND_TYPE_FP_FAST_MATH_MODE:
    case SPV_OPERAND_TYPE_FUNCTION_CONTROL:
    case SPV_OPERAND_TYPE_LOOP_CONTROL:
    case SPV_OPERAND_TYPE_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS:
    case SPV_OPERAND_TYPE_SELECTION_CONTROL:
      return 4;
    default:
      return 0;
  }
  return 0;
}

// Defines and returns current MARK-V version.
uint32_t GetMarkvVersion() {
  const uint32_t kVersionMajor = 1;
  const uint32_t kVersionMinor = 0;
  return kVersionMinor | (kVersionMajor << 16);
}

class CommentLogger {
 public:
  void AppendText(const std::string& str) {
    Append(str);
    delimiter_ = false;
  }

  void AppendTextNewLine(const std::string& str) {
    Append(str);
    Append("\n");
    delimiter_ = false;
  }

  void AppendBitSequence(const std::string& str) {
    if (delimiter_)
      Append("-");
    Append(str);
    delimiter_ = true;
  }

  void AppendWhitespaces(size_t num) {
    Append(std::string(num, ' '));
    delimiter_ = false;
  }

  void NewLine() {
    Append("\n");
    delimiter_ = false;
  }

  std::string GetText() const {
    return ss_.str();
  }

 private:
  void Append(const std::string& str) {
    ss_ << str;
    // std::cerr << str;
  }

  std::stringstream ss_;
  bool delimiter_ = false;
};

spv_text CreateSpvText(const std::string& str) {
  spv_text out = new spv_text_t();
  assert(out);
  char* cstr = new char[str.length() + 1];
  assert(cstr);
  std::strncpy(cstr, str.c_str(), str.length());
  cstr[str.length()] = '\0';
  out->str = cstr;
  out->length = str.length();
  return out;
}

// Base class for MARK-V encoder and decoder.
class MarkvCodecBase {
 public:
  virtual ~MarkvCodecBase() {}

  MarkvCodecBase() = delete;

  void SetModel(const MarkvModel* model) {
    model_ = model;
  }

 protected:
  struct MarkvHeader {
    MarkvHeader() {
      magic_number = kMarkvMagicNumber;
      markv_version = GetMarkvVersion();
      markv_model = 0;
      markv_length_in_bits = 0;
      spirv_version = 0;
      spirv_generator = 0;
    }

    uint32_t magic_number;
    uint32_t markv_version;
    uint32_t markv_model;
    uint32_t markv_length_in_bits;
    uint32_t spirv_version;
    uint32_t spirv_generator;
  };

  explicit MarkvCodecBase(spv_const_context context,
                          const spv_validator_options_t& validator_options)
      : validator_options_(validator_options),
        vstate_(context, &validator_options_), grammar_(context),
        model_(GetDefaultModel()) {}

  spv_result_t UpdateValidationState(const spv_parsed_instruction_t& inst) {
    return ValidateInstructionAndUpdateValidationState(&vstate_, &inst);
  }

  // Returns the current instruction (the one last processed by the validator).
  const Instruction& GetCurrentInstruction() const {
    return vstate_.ordered_instructions().back();
  }

  bool OpcodeHasVariableNumberOfOperands(SpvOp opcode) const {
    spv_opcode_desc opcode_desc;
    if (grammar_.lookupOpcode(static_cast<SpvOp>(opcode), &opcode_desc)
        != SPV_SUCCESS) {
      assert(false && "Opcode not described in the grammar");
    }

    if (opcode_desc->numTypes == 0)
      return false;

    return spvOperandIsVariable(
        opcode_desc->operandTypes[opcode_desc->numTypes - 1]);
  }

  const spv_validator_options_t validator_options_;
  ValidationState_t vstate_;
  const libspirv::AssemblyGrammar grammar_;
  MarkvHeader header_;
  const MarkvModel* model_;
};

class MarkvEncoder : public MarkvCodecBase {
 public:
  MarkvEncoder(spv_const_context context,
               spv_const_markv_encoder_options options)
      : MarkvCodecBase(context, GetValidatorOptions(options)),
        options_(options) {
    (void) options_;
  }

  // Writes data from SPIR-V header to MARK-V header.
  spv_result_t EncodeHeader(
      spv_endianness_t /* endian */, uint32_t /* magic */,
      uint32_t version, uint32_t generator, uint32_t id_bound,
      uint32_t /* schema */) {
    vstate_.setIdBound(id_bound);
    header_.spirv_version = version;
    header_.spirv_generator = generator;
    return SPV_SUCCESS;
  }

  // Encodes SPIR-V instruction to MARK-V and writes to bit stream.
  spv_result_t EncodeInstruction(const spv_parsed_instruction_t& inst);

  // Combines MARK-V header and encoded instructions and the buffer in
  // |markv_binary|.
  void GetMarkvBinary(spv_markv_binary* markv_binary) {
    header_.markv_length_in_bits =
        static_cast<uint32_t>(sizeof(header_) * 8 + writer_.GetNumBits());
    const size_t num_bytes = sizeof(header_) + writer_.GetDataSizeBytes();

    *markv_binary = new spv_markv_binary_t();
    (*markv_binary)->data = new uint8_t[num_bytes];
    (*markv_binary)->length = num_bytes;
    assert(writer_.GetData());
    memcpy((*markv_binary)->data, &header_, sizeof(header_));
    memcpy((*markv_binary)->data + sizeof(header_),
           writer_.GetData(), writer_.GetDataSizeBytes());
  }

  void CreateCommentsLogger() {
    logger_.reset(new CommentLogger());
    writer_.SetCallback([this](const std::string& str){
      logger_->AppendBitSequence(str);
    });
  }

  void AddDisassemblyToComments(std::string&& disassembly) {
    disassembly_.reset(new std::stringstream(std::move(disassembly)));
  }

  void LogDisassemblyInstruction() {
    if (logger_ && disassembly_) {
      std::string line;
      std::getline(*disassembly_, line, '\n');
      logger_->AppendTextNewLine(line);
    }
  }

  std::string GetComments() const {
    if (!logger_)
      return "";
    return logger_->GetText();
  }

 private:
  static spv_validator_options_t GetValidatorOptions(
      spv_const_markv_encoder_options) {
    return spv_validator_options_t();
  }

  // Writes a single word to bit stream. |type| determines if the word is
  // encoded and how.
  void EncodeOperandWord(spv_operand_type_t type, uint32_t word) {
    const size_t chunk_length =
        GetOperandVariableWidthChunkLength(type);
    if (chunk_length) {
      writer_.WriteVariableWidthU32(word, chunk_length);
    } else {
      writer_.WriteUnencoded(word);
    }
  }

  void EncodeLiteralNumber(const Instruction& instruction,
                           const spv_parsed_operand_t& operand);

  spv_const_markv_encoder_options options_;

  // Bit stream where encoded instructions are written.
  BitWriterWord64 writer_;

  std::unique_ptr<CommentLogger> logger_;
  std::unique_ptr<std::stringstream> disassembly_;
};

class MarkvDecoder : public MarkvCodecBase {
 public:
  MarkvDecoder(spv_const_context context,
               const uint8_t* markv_data,
               size_t markv_size_bytes,
               spv_const_markv_decoder_options options)
      : MarkvCodecBase(context, GetValidatorOptions(options)),
        options_(options), reader_(markv_data, markv_size_bytes) {
    (void) options_;
    vstate_.setIdBound(1);
    parsed_operands_.reserve(25);
  }

  // Decodes SPIR-V from MARK-V and stores the words in |spirv_binary|.
  spv_result_t DecodeModule(std::vector<uint32_t>* spirv_binary);

 private:
  // Describes the format of a typed literal number.
  struct NumberType {
    spv_number_kind_t type;
    uint32_t bit_width;
  };

  static spv_validator_options_t GetValidatorOptions(
      spv_const_markv_decoder_options) {
    return spv_validator_options_t();
  }

  // Reads a single word from bit stream. |type| determines if the word needs
  // to be decoded and how.
  bool DecodeOperandWord(spv_operand_type_t type, uint32_t* word) {
    const size_t chunk_length =
        GetOperandVariableWidthChunkLength(type);
    if (chunk_length) {
      return reader_.ReadVariableWidthU32(word, chunk_length);
    } else {
      return reader_.ReadUnencoded(word);
    }
  }

  spv_result_t DecodeLiteralNumber(const spv_parsed_operand_t& operand);

  // Reads instruction from bit stream, decodes and validates it.
  // Decoded instruction is only valid while spirv_ and parsed_operands_ are
  // not changed.
  spv_result_t DecodeInstruction(spv_parsed_instruction_t* inst);

  // Read operand from the stream decodes and validates it.
  spv_result_t DecodeOperand(size_t operand_offset,
                             spv_parsed_instruction_t* inst,
                             const spv_operand_type_t type,
                             spv_operand_pattern_t* expected_operands);

  // Records the numeric type for an operand according to the type information
  // associated with the given non-zero type Id.  This can fail if the type Id
  // is not a type Id, or if the type Id does not reference a scalar numeric
  // type.  On success, return SPV_SUCCESS and populates the num_words,
  // number_kind, and number_bit_width fields of parsed_operand.
  spv_result_t SetNumericTypeInfoForType(spv_parsed_operand_t* parsed_operand,
                                         uint32_t type_id);

  // Records the number type for an instruction at the given offset, if that
  // instruction generates a type.  For types that aren't scalar numbers,
  // record something with number kind SPV_NUMBER_NONE.
  void RecordNumberType(const spv_parsed_instruction_t& inst);

  spv_const_markv_decoder_options options_;

  // Temporaty sink where decoded SPIR-V words are written.
  std::vector<uint32_t> spirv_;

  // Bit stream containing encoded data.
  BitReaderWord64 reader_;

  // Temporary storage for operands of the currently parsed instruction.
  std::vector<spv_parsed_operand_t> parsed_operands_;

  // Maps a result ID to its type ID.  By convention:
  //  - a result ID that is a type definition maps to itself.
  //  - a result ID without a type maps to 0.  (E.g. for OpLabel)
  std::unordered_map<uint32_t, uint32_t> id_to_type_id_;
  // Maps a type ID to its number type description.
  std::unordered_map<uint32_t, NumberType> type_id_to_number_type_info_;
};

void MarkvEncoder::EncodeLiteralNumber(const Instruction& instruction,
                                       const spv_parsed_operand_t& operand) {
  if (operand.number_bit_width == 32) {
    const uint32_t word = instruction.word(operand.offset);
    if (operand.number_kind == SPV_NUMBER_UNSIGNED_INT) {
      writer_.WriteVariableWidthU32(word, model_->u32_chunk_length());
    } else if (operand.number_kind == SPV_NUMBER_SIGNED_INT) {
      int32_t val = 0;
      memcpy(&val, &word, 4);
      writer_.WriteVariableWidthS32(val, model_->s32_chunk_length(),
                                    model_->s32_block_exponent());
    } else if (operand.number_kind == SPV_NUMBER_FLOATING) {
      writer_.WriteUnencoded(word);
    } else {
      assert(0);
    }
  } else if (operand.number_bit_width == 16) {
    const uint16_t word =
        static_cast<uint16_t>(instruction.word(operand.offset));
    if (operand.number_kind == SPV_NUMBER_UNSIGNED_INT) {
      writer_.WriteVariableWidthU16(word, model_->u16_chunk_length());
    } else if (operand.number_kind == SPV_NUMBER_SIGNED_INT) {
      int16_t val = 0;
      memcpy(&val, &word, 2);
      writer_.WriteVariableWidthS16(val, model_->s16_chunk_length(),
                                    model_->s16_block_exponent());
    } else if (operand.number_kind == SPV_NUMBER_FLOATING) {
      writer_.WriteUnencoded(word);
    } else {
      assert(0);
    }
  } else {
    assert(operand.number_bit_width == 64);
    const uint64_t word =
        uint64_t(instruction.word(operand.offset)) |
        (uint64_t(instruction.word(operand.offset + 1)) << 32);
    if (operand.number_kind == SPV_NUMBER_UNSIGNED_INT) {
      writer_.WriteVariableWidthU64(word, model_->u64_chunk_length());
    } else if (operand.number_kind == SPV_NUMBER_SIGNED_INT) {
      int64_t val = 0;
      memcpy(&val, &word, 8);
      writer_.WriteVariableWidthS64(val, model_->s64_chunk_length(),
                                    model_->s64_block_exponent());
    } else if (operand.number_kind == SPV_NUMBER_FLOATING) {
      writer_.WriteUnencoded(word);
    } else {
      assert(0);
    }
  }
}

spv_result_t MarkvEncoder::EncodeInstruction(
    const spv_parsed_instruction_t& inst) {
  const spv_result_t validation_result = UpdateValidationState(inst);
  if (validation_result != SPV_SUCCESS)
    return validation_result;

  assert(vstate_.unresolved_forward_id_count() == 0 &&
         "Forward declared id support not yer implemented");

  const Instruction& instruction = GetCurrentInstruction();
  const auto& operands = instruction.operands();

  LogDisassemblyInstruction();

  // Write opcode.
  writer_.WriteVariableWidthU16(inst.opcode, model_->opcode_chunk_length());

  if (OpcodeHasVariableNumberOfOperands(SpvOp(inst.opcode))) {
    // Write num_operands.
    if (logger_)
      logger_->AppendWhitespaces(kCommentNumWhitespaces);

    writer_.WriteVariableWidthU16(inst.num_operands,
                                  model_->num_operands_chunk_length());
  }

  // Write operands.
  for (const auto& operand : operands) {
    if (operand.type == SPV_OPERAND_TYPE_RESULT_ID) {
      // Result ids are not written (will be regenerated by the decoder).
      // TODO: Support forward declared ids.
      continue;
    }

    if (logger_)
      logger_->AppendWhitespaces(kCommentNumWhitespaces);

    if (operand.type == SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER) {
      EncodeLiteralNumber(instruction, operand);
    } else if (operand.type == SPV_OPERAND_TYPE_LITERAL_STRING) {
      const char* src =
          reinterpret_cast<const char*>(&instruction.words()[operand.offset]);
      const size_t length = spv_strnlen_s(src, operand.num_words * 4);
      if (length == operand.num_words * 4)
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to find terminal character of literal string";
      for (size_t i = 0; i < length + 1; ++i)
        writer_.WriteUnencoded(src[i]);
    } else {
      for (int i = 0; i < operand.num_words; ++i) {
        const uint32_t word = instruction.word(operand.offset + i);
        EncodeOperandWord(operand.type, word);
      }
    }
  }

  if (logger_) {
    logger_->NewLine();
    logger_->NewLine();
  }

  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeLiteralNumber(
    const spv_parsed_operand_t& operand) {
  if (operand.number_bit_width == 32) {
    uint32_t word = 0;
    if (operand.number_kind == SPV_NUMBER_UNSIGNED_INT) {
      if (!reader_.ReadVariableWidthU32(&word, model_->u32_chunk_length()))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to read literal U32";
    } else if (operand.number_kind == SPV_NUMBER_SIGNED_INT) {
      int32_t val = 0;
      if (!reader_.ReadVariableWidthS32(&val, model_->s32_chunk_length(),
                                        model_->s32_block_exponent()))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to read literal S32";
      memcpy(&word, &val, 4);
    } else if (operand.number_kind == SPV_NUMBER_FLOATING) {
      if (!reader_.ReadUnencoded(&word))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to read literal F32";
    } else {
      assert(0);
    }
    spirv_.push_back(word);
  } else if (operand.number_bit_width == 16) {
    uint32_t word = 0;
    if (operand.number_kind == SPV_NUMBER_UNSIGNED_INT) {
      uint16_t val = 0;
      if (!reader_.ReadVariableWidthU16(&val, model_->u16_chunk_length()))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to read literal U16";
      word = val;
    } else if (operand.number_kind == SPV_NUMBER_SIGNED_INT) {
      int16_t val = 0;
      if (!reader_.ReadVariableWidthS16(&val, model_->s16_chunk_length(),
                                        model_->s16_block_exponent()))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to read literal S16";
      // Int16 is stored as int32 in SPIR-V, not as bits.
      int32_t val32 = val;
      memcpy(&word, &val32, 4);
    } else if (operand.number_kind == SPV_NUMBER_FLOATING) {
      uint16_t word16 = 0;
      if (!reader_.ReadUnencoded(&word16))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to read literal F16";
      word = word16;
    } else {
      assert(0);
    }
    spirv_.push_back(word);
  } else {
    assert(operand.number_bit_width == 64);
    uint64_t word = 0;
    if (operand.number_kind == SPV_NUMBER_UNSIGNED_INT) {
      if (!reader_.ReadVariableWidthU64(&word, model_->u64_chunk_length()))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to read literal U64";
    } else if (operand.number_kind == SPV_NUMBER_SIGNED_INT) {
      int64_t val = 0;
      if (!reader_.ReadVariableWidthS64(&val, model_->s64_chunk_length(),
                                        model_->s64_block_exponent()))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to read literal S64";
      memcpy(&word, &val, 8);
    } else if (operand.number_kind == SPV_NUMBER_FLOATING) {
      if (!reader_.ReadUnencoded(&word))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to read literal F64";
    } else {
      assert(0);
    }
    spirv_.push_back(static_cast<uint32_t>(word));
    spirv_.push_back(static_cast<uint32_t>(word >> 32));
  }
  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeModule(std::vector<uint32_t>* spirv_binary) {
  const bool header_read_success =
      reader_.ReadUnencoded(&header_.magic_number) &&
      reader_.ReadUnencoded(&header_.markv_version) &&
      reader_.ReadUnencoded(&header_.markv_model) &&
      reader_.ReadUnencoded(&header_.markv_length_in_bits) &&
      reader_.ReadUnencoded(&header_.spirv_version) &&
      reader_.ReadUnencoded(&header_.spirv_generator);

  if (!header_read_success)
    return vstate_.diag(SPV_ERROR_INVALID_BINARY)
        << "Unable to read MARK-V header";

  assert(header_.magic_number == kMarkvMagicNumber);
  assert(header_.markv_length_in_bits > 0);

  if (header_.magic_number != kMarkvMagicNumber)
    return vstate_.diag(SPV_ERROR_INVALID_BINARY)
        << "MARK-V binary has incorrect magic number";

  // TODO: Print version strings.
  if (header_.markv_version != GetMarkvVersion())
    return vstate_.diag(SPV_ERROR_INVALID_BINARY)
        << "MARK-V binary and the codec have different versions";

  spirv_.reserve(header_.markv_length_in_bits / 2); // Heuristic.
  spirv_.resize(5, 0);
  spirv_[0] = kSpirvMagicNumber;
  spirv_[1] = header_.spirv_version;
  spirv_[2] = header_.spirv_generator;

  while (reader_.GetNumReadBits() < header_.markv_length_in_bits) {
    spv_parsed_instruction_t inst = {};
    const spv_result_t decode_result = DecodeInstruction(&inst);
    if (decode_result != SPV_SUCCESS)
      return decode_result;

    std::cerr << "Instruction decoded, will attempt validation" << std::endl;
    const spv_result_t validation_result = UpdateValidationState(inst);
    if (validation_result != SPV_SUCCESS)
      return validation_result;
    std::cerr << "Instruction validated successfully" << std::endl;
  }

  std::cerr << "Instructions decoding complete" << std::endl;

  if (reader_.GetNumReadBits() != header_.markv_length_in_bits ||
      !reader_.OnlyZeroesLeft()) {
    return vstate_.diag(SPV_ERROR_INVALID_BINARY)
        << "MARK-V binary has wrong stated bit length "
        << reader_.GetNumReadBits() << " " << header_.markv_length_in_bits;
  }

  std::cerr << "Setting id bound" << std::endl;
  spirv_[3] = vstate_.getIdBound();

  *spirv_binary = std::move(spirv_);
  return SPV_SUCCESS;
}

// TODO: The implementation borrows heavily from Parser::parseOperand.
// Consider coupling them together in some way once MARK-V codec is more mature.
// For now it's better to keep the code independent for experimentation
// purposes.
spv_result_t MarkvDecoder::DecodeOperand(
    size_t operand_offset, spv_parsed_instruction_t* inst,
    const spv_operand_type_t type, spv_operand_pattern_t* expected_operands) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);

  spv_parsed_operand_t parsed_operand;
  memset(&parsed_operand, 0, sizeof(parsed_operand));

  assert((operand_offset >> 16) == 0);
  parsed_operand.offset = static_cast<uint16_t>(operand_offset);
  parsed_operand.type = type;

  // Set default values, may be updated later.
  parsed_operand.number_kind = SPV_NUMBER_NONE;
  parsed_operand.number_bit_width = 0;

  const size_t first_word_index = spirv_.size();

  switch (type) {
    case SPV_OPERAND_TYPE_TYPE_ID: {
      if (!DecodeOperandWord(type, &inst->type_id))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to read type_id";

      if (inst->type_id == 0)
        return vstate_.diag(SPV_ERROR_INVALID_BINARY) << "Decoded type_id is 0";

      spirv_.push_back(inst->type_id);
      vstate_.setIdBound(std::max(vstate_.getIdBound(), inst->type_id + 1));
      break;
    }

    case SPV_OPERAND_TYPE_RESULT_ID: {
      inst->result_id = vstate_.getIdBound();
      spirv_.push_back(inst->result_id);
      vstate_.setIdBound(inst->result_id + 1);

      // Save the result ID to type ID mapping.
      // In the grammar, type ID always appears before result ID.
      // A regular value maps to its type. Some instructions (e.g. OpLabel)
      // have no type Id, and will map to 0. The result Id for a
      // type-generating instruction (e.g. OpTypeInt) maps to itself.
      auto insertion_result = id_to_type_id_.emplace(
          inst->result_id,
          spvOpcodeGeneratesType(opcode) ? inst->result_id : inst->type_id);
      if(!insertion_result.second) {
        return vstate_.diag(SPV_ERROR_INVALID_ID)
            << "Unexpected behavior: id->type_id pair was already registered";
      }
      break;
    }

    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_OPTIONAL_ID:
    case SPV_OPERAND_TYPE_SCOPE_ID:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID: {
      uint32_t id = 0;
      if (!DecodeOperandWord(type, &id))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY) << "Failed to read id";

      if (id == 0)
        return vstate_.diag(SPV_ERROR_INVALID_BINARY) << "Decoded id is 0";

      spirv_.push_back(id);
      vstate_.setIdBound(std::max(vstate_.getIdBound(), id + 1));

      if (type == SPV_OPERAND_TYPE_ID || type == SPV_OPERAND_TYPE_OPTIONAL_ID) {

        parsed_operand.type = SPV_OPERAND_TYPE_ID;

        if (opcode == SpvOpExtInst && parsed_operand.offset == 3) {
          assert(0);
        }
      }
      break;
    }

    case SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER: {
      assert(0);
      break;
    }

    case SPV_OPERAND_TYPE_LITERAL_INTEGER:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER: {
      // These are regular single-word literal integer operands.
      // Post-parsing validation should check the range of the parsed value.
      parsed_operand.type = SPV_OPERAND_TYPE_LITERAL_INTEGER;
      // It turns out they are always unsigned integers!
      parsed_operand.number_kind = SPV_NUMBER_UNSIGNED_INT;
      parsed_operand.number_bit_width = 32;

      uint32_t word = 0;
      if (!DecodeOperandWord(type, &word))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to read literal integer";

      spirv_.push_back(word);
      break;
    }

    case SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER:
    case SPV_OPERAND_TYPE_OPTIONAL_TYPED_LITERAL_INTEGER:
      parsed_operand.type = SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER;
      if (opcode == SpvOpSwitch) {
        assert(0);
      } else {
        assert(opcode == SpvOpConstant || opcode == SpvOpSpecConstant);
        // The literal number type is determined by the type Id for the
        // constant.
        assert(inst->type_id);
        if (auto error =
            SetNumericTypeInfoForType(&parsed_operand, inst->type_id))
          return error;

        if (auto error = DecodeLiteralNumber(parsed_operand))
          return error;
      }
      break;

    case SPV_OPERAND_TYPE_LITERAL_STRING:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_STRING: {
      parsed_operand.type = SPV_OPERAND_TYPE_LITERAL_STRING;
      std::vector<char> str;
      while (true) {
        char ch = 0;
        if (!reader_.ReadUnencoded(&ch))
          return vstate_.diag(SPV_ERROR_INVALID_BINARY)
              << "Failed to read literal string";

        str.push_back(ch);

        if (ch == '\0')
          break;
      }

      while (str.size() % 4 != 0)
        str.push_back('\0');

      spirv_.resize(spirv_.size() + str.size() / 4);
      memcpy(&spirv_[first_word_index], str.data(), str.size());

      if (SpvOpExtInstImport == opcode) {
        assert(0);
      }
      break;
    }

    case SPV_OPERAND_TYPE_CAPABILITY:
    case SPV_OPERAND_TYPE_SOURCE_LANGUAGE:
    case SPV_OPERAND_TYPE_EXECUTION_MODEL:
    case SPV_OPERAND_TYPE_ADDRESSING_MODEL:
    case SPV_OPERAND_TYPE_MEMORY_MODEL:
    case SPV_OPERAND_TYPE_EXECUTION_MODE:
    case SPV_OPERAND_TYPE_STORAGE_CLASS:
    case SPV_OPERAND_TYPE_DIMENSIONALITY:
    case SPV_OPERAND_TYPE_SAMPLER_ADDRESSING_MODE:
    case SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE:
    case SPV_OPERAND_TYPE_SAMPLER_IMAGE_FORMAT:
    case SPV_OPERAND_TYPE_FP_ROUNDING_MODE:
    case SPV_OPERAND_TYPE_LINKAGE_TYPE:
    case SPV_OPERAND_TYPE_ACCESS_QUALIFIER:
    case SPV_OPERAND_TYPE_OPTIONAL_ACCESS_QUALIFIER:
    case SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE:
    case SPV_OPERAND_TYPE_DECORATION:
    case SPV_OPERAND_TYPE_BUILT_IN:
    case SPV_OPERAND_TYPE_GROUP_OPERATION:
    case SPV_OPERAND_TYPE_KERNEL_ENQ_FLAGS:
    case SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO: {
      // A single word that is a plain enum value.
      uint32_t word = 0;
      if (!DecodeOperandWord(type, &word))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to read literal integer";

      spirv_.push_back(word);

      // Map an optional operand type to its corresponding concrete type.
      if (type == SPV_OPERAND_TYPE_OPTIONAL_ACCESS_QUALIFIER)
        parsed_operand.type = SPV_OPERAND_TYPE_ACCESS_QUALIFIER;

      spv_operand_desc entry;
      if (grammar_.lookupOperand(type, word, &entry)) {
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Invalid "
            << spvOperandTypeStr(parsed_operand.type)
            << " operand: " << word;
      }

      // Prepare to accept operands to this operand, if needed.
      spvPrependOperandTypes(entry->operandTypes, expected_operands);
      break;
    }

    case SPV_OPERAND_TYPE_FP_FAST_MATH_MODE:
    case SPV_OPERAND_TYPE_FUNCTION_CONTROL:
    case SPV_OPERAND_TYPE_LOOP_CONTROL:
    case SPV_OPERAND_TYPE_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS:
    case SPV_OPERAND_TYPE_SELECTION_CONTROL: {
      // This operand is a mask.
      uint32_t word = 0;
      if (!DecodeOperandWord(type, &word))
        return vstate_.diag(SPV_ERROR_INVALID_BINARY)
            << "Failed to read " << spvOperandTypeStr(type)
            << " for " << spvOpcodeString(SpvOp(inst->opcode));

      spirv_.push_back(word);

      // Map an optional operand type to its corresponding concrete type.
      if (type == SPV_OPERAND_TYPE_OPTIONAL_IMAGE)
        parsed_operand.type = SPV_OPERAND_TYPE_IMAGE;
      else if (type == SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS)
        parsed_operand.type = SPV_OPERAND_TYPE_MEMORY_ACCESS;

      // Check validity of set mask bits. Also prepare for operands for those
      // masks if they have any.  To get operand order correct, scan from
      // MSB to LSB since we can only prepend operands to a pattern.
      // The only case in the grammar where you have more than one mask bit
      // having an operand is for image operands.  See SPIR-V 3.14 Image
      // Operands.
      uint32_t remaining_word = word;
      for (uint32_t mask = (1u << 31); remaining_word; mask >>= 1) {
        if (remaining_word & mask) {
          spv_operand_desc entry;
          if (grammar_.lookupOperand(type, mask, &entry)) {
            return vstate_.diag(SPV_ERROR_INVALID_BINARY)
                   << "Invalid " << spvOperandTypeStr(parsed_operand.type)
                   << " operand: " << word << " has invalid mask component "
                   << mask;
          }
          remaining_word ^= mask;
          spvPrependOperandTypes(entry->operandTypes, expected_operands);
        }
      }
      if (word == 0) {
        // An all-zeroes mask *might* also be valid.
        spv_operand_desc entry;
        if (SPV_SUCCESS == grammar_.lookupOperand(type, 0, &entry)) {
          // Prepare for its operands, if any.
          spvPrependOperandTypes(entry->operandTypes, expected_operands);
        }
      }
      break;
    }
    default:
      return vstate_.diag(SPV_ERROR_INVALID_BINARY)
          << "Internal error: Unhandled operand type: " << type;
  }

  parsed_operand.num_words = uint16_t(spirv_.size() - first_word_index);

  assert(int(SPV_OPERAND_TYPE_FIRST_CONCRETE_TYPE) <= int(parsed_operand.type));
  assert(int(SPV_OPERAND_TYPE_LAST_CONCRETE_TYPE) >= int(parsed_operand.type));

  parsed_operands_.push_back(parsed_operand);

  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeInstruction(spv_parsed_instruction_t* inst) {
  parsed_operands_.clear();
  const size_t inst_module_offset = spirv_.size();

  if (!reader_.ReadVariableWidthU16(&inst->opcode,
                                    model_->opcode_chunk_length())) {
    return vstate_.diag(SPV_ERROR_INVALID_BINARY)
        << "Failed to read opcode of instruction";
  }

  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);

  std::cerr << "\nOpcode: " << spvOpcodeString(opcode) << std::endl;

  // Opcode/num_words placeholder, the word will be filled in later.
  spirv_.push_back(0);

  const size_t first_operand_module_offset = spirv_.size();

  spv_opcode_desc opcode_desc;
  if (grammar_.lookupOpcode(opcode, &opcode_desc)
      != SPV_SUCCESS) {
    return vstate_.diag(SPV_ERROR_INVALID_BINARY) << "Invalid opcode";
  }

  spv_operand_pattern_t expected_operands(
      opcode_desc->operandTypes,
      opcode_desc->operandTypes + opcode_desc->numTypes);

  if (OpcodeHasVariableNumberOfOperands(opcode)) {
    if (!reader_.ReadVariableWidthU16(&inst->num_operands,
                                      model_->num_operands_chunk_length()))
      return vstate_.diag(SPV_ERROR_INVALID_BINARY)
          << "Failed to read num_operands of instruction";
  } else {
    inst->num_operands = static_cast<uint16_t>(expected_operands.size());
  }

  std::cerr << "Num operands: " << inst->num_operands << std::endl;

  for (size_t operand_index = 0;
       operand_index < static_cast<size_t>(inst->num_operands);
       ++operand_index) {
    assert(!expected_operands.empty());
    const spv_operand_type_t type =
        spvTakeFirstMatchableOperand(&expected_operands);

    const size_t operand_offset = spirv_.size() - first_operand_module_offset;

    const spv_result_t decode_result =
        DecodeOperand(operand_offset, inst, type, &expected_operands);

    std::cerr << "operand" << operand_index << ": "
              << spvOperandTypeStr(parsed_operands_.back().type) << std::endl;

    if (decode_result != SPV_SUCCESS)
      return decode_result;
  }

  std::cerr << "Operands decoded successfully" << std::endl;

  assert(inst->num_operands == parsed_operands_.size());

  // Only valid while spirv_ and parsed_operands_ remain unchanged.
  std::cerr << "Setting inst->words" << std::endl;
  inst->words = &spirv_[first_operand_module_offset];
  std::cerr << "Setting inst->operands" << std::endl;
  inst->operands = parsed_operands_.empty() ? nullptr : parsed_operands_.data();
  std::cerr << "Setting inst->num_words" << std::endl;
  inst->num_words = static_cast<uint16_t>(spirv_.size() - inst_module_offset);
  std::cerr << "Writing to spirv_[inst_module_offset]" << std::endl;
  spirv_[inst_module_offset] =
      spvOpcodeMake(inst->num_words, SpvOp(inst->opcode));

  std::cerr << "Parsed instruction created" << std::endl;

  assert(inst->num_words == std::accumulate(
      parsed_operands_.begin(), parsed_operands_.end(), 1,
      [](size_t num_words, const spv_parsed_operand_t& operand) {
        return num_words += operand.num_words;
  }) && "num_words in instruction doesn't correspond to the sum of num_words"
        "in the operands");

  std::cerr << "Calling Record number type" << std::endl;
  RecordNumberType(*inst);
  std::cerr << "Exiting DecodeInstruction" << std::endl;

  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::SetNumericTypeInfoForType(
    spv_parsed_operand_t* parsed_operand, uint32_t type_id) {
  assert(type_id != 0);
  auto type_info_iter = type_id_to_number_type_info_.find(type_id);
  if (type_info_iter == type_id_to_number_type_info_.end()) {
    return vstate_.diag(SPV_ERROR_INVALID_BINARY)
        << "Type Id " << type_id << " is not a type";
  }

  const NumberType& info = type_info_iter->second;
  if (info.type == SPV_NUMBER_NONE) {
    // This is a valid type, but for something other than a scalar number.
    return vstate_.diag(SPV_ERROR_INVALID_BINARY)
        << "Type Id " << type_id << " is not a scalar numeric type";
  }

  parsed_operand->number_kind = info.type;
  parsed_operand->number_bit_width = info.bit_width;
  // Round up the word count.
  parsed_operand->num_words = static_cast<uint16_t>((info.bit_width + 31) / 32);
  return SPV_SUCCESS;
}

void MarkvDecoder::RecordNumberType(const spv_parsed_instruction_t& inst) {
  const SpvOp opcode = static_cast<SpvOp>(inst.opcode);
  if (spvOpcodeGeneratesType(opcode)) {
    NumberType info = {SPV_NUMBER_NONE, 0};
    if (SpvOpTypeInt == opcode) {
      info.bit_width = inst.words[inst.operands[1].offset];
      info.type = inst.words[inst.operands[2].offset] ?
          SPV_NUMBER_SIGNED_INT : SPV_NUMBER_UNSIGNED_INT;
    } else if (SpvOpTypeFloat == opcode) {
      info.bit_width = inst.words[inst.operands[1].offset];
      info.type = SPV_NUMBER_FLOATING;
    }
    // The *result* Id of a type generating instruction is the type Id.
    type_id_to_number_type_info_[inst.result_id] = info;
  }
}

spv_result_t EncodeHeader(
    void* user_data, spv_endianness_t endian, uint32_t magic,
    uint32_t version, uint32_t generator, uint32_t id_bound,
    uint32_t schema) {
  MarkvEncoder* encoder = reinterpret_cast<MarkvEncoder*>(user_data);
  return encoder->EncodeHeader(
      endian, magic, version, generator, id_bound, schema);
}

spv_result_t EncodeInstruction(
    void* user_data, const spv_parsed_instruction_t* inst) {
  MarkvEncoder* encoder = reinterpret_cast<MarkvEncoder*>(user_data);
  return encoder->EncodeInstruction(*inst);
}

}  // namespace

spv_result_t spvSpirvToMarkv(spv_const_context context,
                             const uint32_t* spirv_words,
                             const size_t spirv_num_words,
                             spv_const_markv_encoder_options options,
                             spv_markv_binary* markv_binary,
                             spv_text* comments, spv_diagnostic* diagnostic) {
  spv_context_t hijack_context = *context;
  if (diagnostic) {
    *diagnostic = nullptr;
    libspirv::UseDiagnosticAsMessageConsumer(&hijack_context, diagnostic);
  }

  spv_const_binary_t spirv_binary = {spirv_words, spirv_num_words};

  spv_endianness_t endian;
  spv_position_t position = {};
  if (spvBinaryEndianness(&spirv_binary, &endian)) {
    return libspirv::DiagnosticStream(position, hijack_context.consumer,
                                      SPV_ERROR_INVALID_BINARY)
        << "Invalid SPIR-V magic number.";
  }

  spv_header_t header;
  if (spvBinaryHeaderGet(&spirv_binary, endian, &header)) {
    return libspirv::DiagnosticStream(position, hijack_context.consumer,
                                      SPV_ERROR_INVALID_BINARY)
        << "Invalid SPIR-V header.";
  }

  MarkvEncoder encoder(&hijack_context, options);

  if (comments) {
    encoder.CreateCommentsLogger();

    spv_text text = nullptr;
    if (spvBinaryToText(&hijack_context, spirv_words, spirv_num_words,
                        SPV_BINARY_TO_TEXT_OPTION_NO_HEADER, &text, nullptr)
        != SPV_SUCCESS) {
      return libspirv::DiagnosticStream(position, hijack_context.consumer,
                                        SPV_ERROR_INVALID_BINARY)
          << "Failed to disassemble SPIR-V binary.";
    }
    assert(text);
    encoder.AddDisassemblyToComments(std::string(text->str, text->length));
    spvTextDestroy(text);
  }

  if (spvBinaryParse(
      &hijack_context, &encoder, spirv_words, spirv_num_words, EncodeHeader,
      EncodeInstruction, diagnostic) != SPV_SUCCESS) {
    return libspirv::DiagnosticStream(position, hijack_context.consumer,
                                      SPV_ERROR_INVALID_BINARY)
        << "Unable to encode to MARK-V.";
  }

  if (comments)
    *comments = CreateSpvText(encoder.GetComments());

  encoder.GetMarkvBinary(markv_binary);
  return SPV_SUCCESS;
}

spv_result_t spvMarkvToSpirv(spv_const_context context,
                             const uint8_t* markv_data,
                             size_t markv_size_bytes,
                             spv_const_markv_decoder_options options,
                             spv_binary* spirv_binary,
                             spv_text* /* comments */, spv_diagnostic* diagnostic) {
  spv_position_t position = {};
  spv_context_t hijack_context = *context;
  if (diagnostic) {
    *diagnostic = nullptr;
    libspirv::UseDiagnosticAsMessageConsumer(&hijack_context, diagnostic);
  }

  MarkvDecoder decoder(&hijack_context, markv_data, markv_size_bytes, options);

  std::vector<uint32_t> words;

  if (decoder.DecodeModule(&words) != SPV_SUCCESS) {
    return libspirv::DiagnosticStream(position, hijack_context.consumer,
                                      SPV_ERROR_INVALID_BINARY)
        << "Unable to decode MARK-V.";
  }

  assert(!words.empty());

  std::cerr << "Creating spirv_binary" << std::endl;
  *spirv_binary = new spv_binary_t();
  (*spirv_binary)->code = new uint32_t[words.size()];
  (*spirv_binary)->wordCount = words.size();
  memcpy((*spirv_binary)->code, words.data(), 4 * words.size());

  std::cerr << "spvMarkvToSpirv finished successfully" << std::endl;
  return SPV_SUCCESS;
}

void spvMarkvBinaryDestroy(spv_markv_binary binary) {
  if (!binary) return;
  delete[] binary->data;
  delete binary;
}
