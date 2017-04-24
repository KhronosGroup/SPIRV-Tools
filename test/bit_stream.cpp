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

#include <sstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "util/bit_stream.h"

namespace {

using spvutils::BitWriterInterface;
using spvutils::BitReaderInterface;
using spvutils::BitWriterWord64;
using spvutils::BitReaderWord64;
using spvutils::StreamToBuffer;
using spvutils::BufferToStream;
using spvutils::NumBitsToNumWords;
using spvutils::PadToWord;
using spvutils::StreamToBitset;
using spvutils::BitsetToStream;
using spvutils::BitsToStream;
using spvutils::StreamToBits;
using spvutils::GetLowerBits;
using spvutils::EncodeZigZag;
using spvutils::DecodeZigZag;

// A simple and inefficient implementatition of BitWriterInterface,
// using std::stringstream. Intended for tests only.
class BitWriterStringStream : public BitWriterInterface {
 public:
  void WriteStream(const std::string& bits) override {
    ss_ << bits;
  }

  void WriteBits(uint64_t bits, size_t num_bits) override {
    assert(num_bits <= 64);
    ss_ << BitsToStream(bits, num_bits);
  }

  size_t GetNumBits() const override {
    return ss_.str().size();
  }

  std::vector<uint8_t> GetDataCopy() const override {
    return StreamToBuffer<uint8_t>(ss_.str());
  }

  std::string GetStreamRaw() const {
    return ss_.str();
  }

 private:
  std::stringstream ss_;
};

// A simple and inefficient implementatition of BitReaderInterface.
// Intended for tests only.
class BitReaderFromString : public BitReaderInterface {
 public:
  explicit BitReaderFromString(std::string&& str)
      : str_(std::move(str)), pos_(0) {}

  explicit BitReaderFromString(const std::vector<uint64_t>& buffer)
      : str_(BufferToStream(buffer)), pos_(0) {}

  explicit BitReaderFromString(const std::vector<uint8_t>& buffer)
      : str_(PadToWord<64>(BufferToStream(buffer))), pos_(0) {}

  size_t ReadBits(uint64_t* bits, size_t num_bits) override {
    if (ReachedEnd())
      return 0;
    std::string sub = str_.substr(pos_, num_bits);
    *bits = StreamToBits(sub);
    pos_ += sub.length();
    return sub.length();
  }

  bool ReachedEnd() const override {
    return pos_ >= str_.length();
  }

  const std::string& GetStreamPadded64() const {
    return str_;
  }

 private:
  std::string str_;
  size_t pos_;
};

TEST(ZigZag, Encode) {
  EXPECT_EQ(0, EncodeZigZag(0));
  EXPECT_EQ(1, EncodeZigZag(-1));
  EXPECT_EQ(2, EncodeZigZag(1));
  EXPECT_EQ(3, EncodeZigZag(-2));
  EXPECT_EQ(4, EncodeZigZag(2));
  EXPECT_EQ(5, EncodeZigZag(-3));
  EXPECT_EQ(6, EncodeZigZag(3));
}

TEST(ZigZag, Decode) {
  EXPECT_EQ(0, DecodeZigZag(0));
  EXPECT_EQ(-1, DecodeZigZag(1));
  EXPECT_EQ(1, DecodeZigZag(2));
  EXPECT_EQ(-2, DecodeZigZag(3));
  EXPECT_EQ(2, DecodeZigZag(4));
  EXPECT_EQ(-3, DecodeZigZag(5));
  EXPECT_EQ(3, DecodeZigZag(6));
}

TEST(BufToStream, UInt8_Empty) {
  const std::string expected_bits = "";
  std::vector<uint8_t> buffer = StreamToBuffer<uint8_t>(expected_bits);
  const std::string result_bits = BufferToStream(buffer);
  EXPECT_EQ(expected_bits, result_bits);
}

TEST(BufToStream, UInt8_OneWord) {
  const std::string expected_bits = "00101100";
  std::vector<uint8_t> buffer = StreamToBuffer<uint8_t>(expected_bits);
  const std::string result_bits = BufferToStream(buffer);
  EXPECT_EQ(expected_bits, result_bits);
}

TEST(BufToStream, UInt8_MultipleWords) {
  const std::string expected_bits = "00100010011010100111110100100010";
  std::vector<uint8_t> buffer = StreamToBuffer<uint8_t>(expected_bits);
  const std::string result_bits = BufferToStream(buffer);
  EXPECT_EQ(expected_bits, result_bits);
}

TEST(BufToStream, UInt64_Empty) {
  const std::string expected_bits = "";
  std::vector<uint64_t> buffer = StreamToBuffer<uint64_t>(expected_bits);
  const std::string result_bits = BufferToStream(buffer);
  EXPECT_EQ(expected_bits, result_bits);
}

TEST(BufToStream, UInt64_OneWord) {
  const std::string expected_bits =
      "0010001001101010011111010010001001001010000111110010010010010101";
  std::vector<uint64_t> buffer = StreamToBuffer<uint64_t>(expected_bits);
  const std::string result_bits = BufferToStream(buffer);
  EXPECT_EQ(expected_bits, result_bits);
}

TEST(BufToStream, UInt64_Unaligned) {
  const std::string expected_bits =
      "0010001001101010011111010010001001001010000111110010010010010101"
      "0010001001101010011111111111111111111111";
  std::vector<uint64_t> buffer = StreamToBuffer<uint64_t>(expected_bits);
  const std::string result_bits = BufferToStream(buffer);
  EXPECT_EQ(PadToWord<64>(expected_bits), result_bits);
}

TEST(BufToStream, UInt64_MultipleWords) {
  const std::string expected_bits =
      "0010001001101010011111010010001001001010000111110010010010010101"
      "0010001001101010011111111111111111111111000111110010010010010111"
      "0000000000000000000000000000000000000000000000000010010011111111";
  std::vector<uint64_t> buffer = StreamToBuffer<uint64_t>(expected_bits);
  const std::string result_bits = BufferToStream(buffer);
  EXPECT_EQ(expected_bits, result_bits);
}

TEST(PadToWord, Test) {
  EXPECT_EQ("10100000", PadToWord<8>("101"));
  EXPECT_EQ("10100000""00000000", PadToWord<16>("101"));
  EXPECT_EQ("10100000""00000000""00000000""00000000",
            PadToWord<32>("101"));
  EXPECT_EQ("10100000""00000000""00000000""00000000"
            "00000000""00000000""00000000""00000000",
            PadToWord<64>("101"));
}

TEST(BitWriterStringStream, Empty) {
  BitWriterStringStream writer;
  EXPECT_EQ(0, writer.GetNumBits());
  EXPECT_EQ(0, writer.GetDataSizeBytes());
  EXPECT_EQ("", writer.GetStreamRaw());
}

TEST(BitWriterStringStream, WriteStream) {
  BitWriterStringStream writer;
  const std::string bits1 = "1011111111111111111";
  writer.WriteStream(bits1);
  EXPECT_EQ(19, writer.GetNumBits());
  EXPECT_EQ(3, writer.GetDataSizeBytes());
  EXPECT_EQ(bits1, writer.GetStreamRaw());

  const std::string bits2 = "10100001010101010000111111111111111111111111111";
  writer.WriteStream(bits2);
  EXPECT_EQ(66, writer.GetNumBits());
  EXPECT_EQ(9, writer.GetDataSizeBytes());
  EXPECT_EQ(bits1 + bits2, writer.GetStreamRaw());
}

TEST(BitWriterStringStream, WriteBitSet) {
  BitWriterStringStream writer;
  const std::string bits1 = "10101";
  writer.WriteBitset(StreamToBitset<16>(bits1));
  EXPECT_EQ(16, writer.GetNumBits());
  EXPECT_EQ(2, writer.GetDataSizeBytes());
  EXPECT_EQ(PadToWord<16>(bits1), writer.GetStreamRaw());
}

TEST(BitWriterStringStream, WriteBits) {
  BitWriterStringStream writer;
  const uint64_t bits1 = 0x1 | 0x2 | 0x10;
  writer.WriteBits(bits1, 5);
  EXPECT_EQ(5, writer.GetNumBits());
  EXPECT_EQ(1, writer.GetDataSizeBytes());
  EXPECT_EQ("11001", writer.GetStreamRaw());
}

TEST(BitWriterStringStream, WriteMultiple) {
  BitWriterStringStream writer;

  std::string expected_result;
  const std::string bits1 = "101001111111001100010000001110001111111100";
  writer.WriteStream(bits1);

  const std::string bits2 = "10100011000010010101";
  writer.WriteBitset(StreamToBitset<20>(bits2));

  const uint64_t val = 0x1 | 0x2 | 0x10;
  const std::string bits3 = BitsToStream(val, 8);
  writer.WriteBits(val, 8);

  const std::string expected = bits1 + bits2 + bits3;

  EXPECT_EQ(expected.length(), writer.GetNumBits());
  EXPECT_EQ(9, writer.GetDataSizeBytes());
  EXPECT_EQ(expected, writer.GetStreamRaw());

  EXPECT_EQ(PadToWord<8>(expected), BufferToStream(writer.GetDataCopy()));
}

TEST(BitWriterWord64, Empty) {
  BitWriterWord64 writer;
  EXPECT_EQ(0, writer.GetNumBits());
  EXPECT_EQ(0, writer.GetDataSizeBytes());
  EXPECT_EQ("", writer.GetStreamPadded64());
}

TEST(BitWriterWord64, WriteStream) {
  BitWriterWord64 writer;
  std::string expected;

  {
    const std::string bits = "101";
    expected += bits;
    writer.WriteStream(bits);
    EXPECT_EQ(expected.length(), writer.GetNumBits());
    EXPECT_EQ(1, writer.GetDataSizeBytes());
    EXPECT_EQ(PadToWord<64>(expected), writer.GetStreamPadded64());
  }

  {
    const std::string bits = "10000111111111110000000";
    expected += bits;
    writer.WriteStream(bits);
    EXPECT_EQ(expected.length(), writer.GetNumBits());
    EXPECT_EQ(PadToWord<64>(expected), writer.GetStreamPadded64());
  }

  {
    const std::string bits = "101001111111111100000111111111111100";
    expected += bits;
    writer.WriteStream(bits);
    EXPECT_EQ(expected.length(), writer.GetNumBits());
    EXPECT_EQ(PadToWord<64>(expected), writer.GetStreamPadded64());
  }
}

TEST(BitWriterWord64, WriteBitset) {
  BitWriterWord64 writer;
  const std::string bits1 = "10101";
  writer.WriteBitset(StreamToBitset<16>(bits1), 12);
  EXPECT_EQ(12, writer.GetNumBits());
  EXPECT_EQ(2, writer.GetDataSizeBytes());
  EXPECT_EQ(PadToWord<64>(bits1), writer.GetStreamPadded64());
}

TEST(BitWriterWord64, WriteBits) {
  BitWriterWord64 writer;
  const uint64_t bits1 = 0x1 | 0x2 | 0x10;
  writer.WriteBits(bits1, 5);
  writer.WriteBits(bits1, 5);
  writer.WriteBits(bits1, 5);
  EXPECT_EQ(15, writer.GetNumBits());
  EXPECT_EQ(2, writer.GetDataSizeBytes());
  EXPECT_EQ(PadToWord<64>("110011100111001"), writer.GetStreamPadded64());
}

TEST(BitWriterWord64, ComparisonTestWriteLotsOfBits) {
  BitWriterStringStream writer1;
  BitWriterWord64 writer2(16384);

  for (uint64_t i = 0; i < 65000; i += 25) {
    writer1.WriteBits(i, 16);
    writer2.WriteBits(i, 16);
    EXPECT_EQ(writer1.GetNumBits(), writer2.GetNumBits());
  }

  EXPECT_EQ(PadToWord<64>(writer1.GetStreamRaw()),
            writer2.GetStreamPadded64());
}

TEST(BitWriterWord64, ComparisonTestWriteLotsOfStreams) {
  BitWriterStringStream writer1;
  BitWriterWord64 writer2(16384);

  for (int i = 0; i < 1000; ++i) {
    std::string bits = "1111100000";
    if (i % 2)
      bits += "101010";
    if (i % 3)
      bits += "1110100";
    if (i % 5)
      bits += "1110100111111111111";
    writer1.WriteStream(bits);
    writer2.WriteStream(bits);
    EXPECT_EQ(writer1.GetNumBits(), writer2.GetNumBits());
  }

  EXPECT_EQ(PadToWord<64>(writer1.GetStreamRaw()),
            writer2.GetStreamPadded64());
}

TEST(BitWriterWord64, ComparisonTestWriteLotsOfBitsets) {
  BitWriterStringStream writer1;
  BitWriterWord64 writer2(16384);

  for (uint64_t i = 0; i < 65000; i += 25) {
    std::bitset<16> bits1(i);
    std::bitset<24> bits2(i);
    writer1.WriteBitset(bits1);
    writer1.WriteBitset(bits2);
    writer2.WriteBitset(bits1);
    writer2.WriteBitset(bits2);
    EXPECT_EQ(writer1.GetNumBits(), writer2.GetNumBits());
  }

  EXPECT_EQ(PadToWord<64>(writer1.GetStreamRaw()),
            writer2.GetStreamPadded64());
}

TEST(GetLowerBits, Test) {
  EXPECT_EQ(0, GetLowerBits<uint8_t>(255, 0));
  EXPECT_EQ(1, GetLowerBits<uint8_t>(255, 1));
  EXPECT_EQ(3, GetLowerBits<uint8_t>(255, 2));
  EXPECT_EQ(7, GetLowerBits<uint8_t>(255, 3));
  EXPECT_EQ(15, GetLowerBits<uint8_t>(255, 4));
  EXPECT_EQ(31, GetLowerBits<uint8_t>(255, 5));
  EXPECT_EQ(63, GetLowerBits<uint8_t>(255, 6));
  EXPECT_EQ(127, GetLowerBits<uint8_t>(255, 7));
  EXPECT_EQ(255, GetLowerBits<uint8_t>(255, 8));
  EXPECT_EQ(0xFF, GetLowerBits<uint32_t>(0xFFFFFFFF, 8));
  EXPECT_EQ(0xFFFF, GetLowerBits<uint32_t>(0xFFFFFFFF, 16));
  EXPECT_EQ(0xFFFFFF, GetLowerBits<uint32_t>(0xFFFFFFFF, 24));
  EXPECT_EQ(0xFFFFFF, GetLowerBits<uint64_t>(0xFFFFFFFFFFFF, 24));
  EXPECT_EQ(0xFFFFFFFFFFFFFFFF,
            GetLowerBits<uint64_t>(0xFFFFFFFFFFFFFFFF, 64));
  EXPECT_EQ(StreamToBits("1010001110"),
            GetLowerBits<uint64_t>(
                StreamToBits("1010001110111101111111"), 10));
}

TEST(BitReaderFromString, FromU8) {
  std::vector<uint8_t> buffer = {
    0xAA, 0xBB, 0xCC, 0xDD,
  };

  const std::string total_stream =
      "01010101""11011101""00110011""10111011";

  BitReaderFromString reader(buffer);
  EXPECT_EQ(PadToWord<64>(total_stream), reader.GetStreamPadded64());

  uint64_t bits = 0;
  EXPECT_EQ(2, reader.ReadBits(&bits, 2));
  EXPECT_EQ(PadToWord<64>("01"), BitsToStream(bits));
  EXPECT_EQ(20, reader.ReadBits(&bits, 20));
  EXPECT_EQ(PadToWord<64>("01010111011101001100"), BitsToStream(bits));
  EXPECT_EQ(20, reader.ReadBits(&bits, 20));
  EXPECT_EQ(PadToWord<64>("11101110110000000000"), BitsToStream(bits));
  EXPECT_EQ(22, reader.ReadBits(&bits, 30));
  EXPECT_EQ(PadToWord<64>("0000000000000000000000"), BitsToStream(bits));
  EXPECT_TRUE(reader.ReachedEnd());
}

TEST(BitReaderFromString, FromU64) {
  std::vector<uint64_t> buffer = {
    0xAAAAAAAAAAAAAAAA,
    0xBBBBBBBBBBBBBBBB,
    0xCCCCCCCCCCCCCCCC,
    0xDDDDDDDDDDDDDDDD,
  };

  const std::string total_stream =
      "0101010101010101010101010101010101010101010101010101010101010101"
      "1101110111011101110111011101110111011101110111011101110111011101"
      "0011001100110011001100110011001100110011001100110011001100110011"
      "1011101110111011101110111011101110111011101110111011101110111011";

  BitReaderFromString reader(buffer);
  EXPECT_EQ(total_stream, reader.GetStreamPadded64());

  uint64_t bits = 0;
  size_t pos = 0;
  size_t to_read = 5;
  while (reader.ReadBits(&bits, to_read) > 0) {
    EXPECT_EQ(BitsToStream(bits),
              PadToWord<64>(total_stream.substr(pos, to_read)));
    pos += to_read;
    to_read = (to_read + 35) % 64 + 1;
  }
  EXPECT_TRUE(reader.ReachedEnd());
}

TEST(BitReaderWord64, ReadBitsSingleByte) {
  BitReaderWord64 reader(std::vector<uint8_t>({0xF0}));
  EXPECT_FALSE(reader.ReachedEnd());

  uint64_t bits = 0;
  EXPECT_EQ(1, reader.ReadBits(&bits, 1));
  EXPECT_EQ(0, bits);
  EXPECT_EQ(2, reader.ReadBits(&bits, 2));
  EXPECT_EQ(0, bits);
  EXPECT_EQ(2, reader.ReadBits(&bits, 2));
  EXPECT_EQ(2, bits);
  EXPECT_EQ(2, reader.ReadBits(&bits, 2));
  EXPECT_EQ(3, bits);
  EXPECT_FALSE(reader.OnlyZeroesLeft());
  EXPECT_FALSE(reader.ReachedEnd());
  EXPECT_EQ(2, reader.ReadBits(&bits, 2));
  EXPECT_EQ(1, bits);
  EXPECT_TRUE(reader.OnlyZeroesLeft());
  EXPECT_FALSE(reader.ReachedEnd());
  EXPECT_EQ(55, reader.ReadBits(&bits, 64));
  EXPECT_EQ(0, bits);
  EXPECT_TRUE(reader.ReachedEnd());
}

TEST(BitReaderWord64, ReadBitsetSingleByte) {
  BitReaderWord64 reader(std::vector<uint8_t>({0xCC}));
  std::bitset<4> bits;
  EXPECT_EQ(2, reader.ReadBitset(&bits, 2));
  EXPECT_EQ(0, bits.to_ullong());
  EXPECT_EQ(2, reader.ReadBitset(&bits, 2));
  EXPECT_EQ(3, bits.to_ullong());
  EXPECT_FALSE(reader.OnlyZeroesLeft());
  EXPECT_EQ(4, reader.ReadBitset(&bits, 4));
  EXPECT_EQ(12, bits.to_ullong());
  EXPECT_TRUE(reader.OnlyZeroesLeft());
}

TEST(BitReaderWord64, ReadStreamSingleByte) {
  BitReaderWord64 reader(std::vector<uint8_t>({0xAA}));
  EXPECT_EQ("", reader.ReadStream(0));
  EXPECT_EQ("0", reader.ReadStream(1));
  EXPECT_EQ("101", reader.ReadStream(3));
  EXPECT_EQ("01010000", reader.ReadStream(8));
  EXPECT_TRUE(reader.OnlyZeroesLeft());
  EXPECT_EQ("0000000000000000000000000000000000000000000000000000",
            reader.ReadStream(64));
  EXPECT_TRUE(reader.ReachedEnd());
}

TEST(BitReaderWord64, ReadStreamEmpty) {
  std::vector<uint64_t> buffer;
  BitReaderWord64 reader(std::move(buffer));
  EXPECT_TRUE(reader.OnlyZeroesLeft());
  EXPECT_TRUE(reader.ReachedEnd());
  EXPECT_EQ("", reader.ReadStream(10));
  EXPECT_TRUE(reader.ReachedEnd());
}

TEST(BitReaderWord64, ReadBitsTwoWords) {
  std::vector<uint64_t> buffer = {
    0x0000000000000001,
    0x0000000000FFFFFF
  };

  BitReaderWord64 reader(std::move(buffer));

  uint64_t bits = 0;
  EXPECT_EQ(1, reader.ReadBits(&bits, 1));
  EXPECT_EQ(1, bits);
  EXPECT_EQ(62, reader.ReadBits(&bits, 62));
  EXPECT_EQ(0, bits);
  EXPECT_EQ(2, reader.ReadBits(&bits, 2));
  EXPECT_EQ(2, bits);
  EXPECT_EQ(3, reader.ReadBits(&bits, 3));
  EXPECT_EQ(7, bits);
  EXPECT_FALSE(reader.OnlyZeroesLeft());
  EXPECT_EQ(32, reader.ReadBits(&bits, 32));
  EXPECT_EQ(0xFFFFF, bits);
  EXPECT_TRUE(reader.OnlyZeroesLeft());
  EXPECT_FALSE(reader.ReachedEnd());
  EXPECT_EQ(28, reader.ReadBits(&bits, 32));
  EXPECT_EQ(0, bits);
  EXPECT_TRUE(reader.ReachedEnd());
}

TEST(BitReaderWord64, FromU8) {
  std::vector<uint8_t> buffer = {
    0xAA, 0xBB, 0xCC, 0xDD,
  };

  BitReaderWord64 reader(std::move(buffer));

  uint64_t bits = 0;
  EXPECT_EQ(2, reader.ReadBits(&bits, 2));
  EXPECT_EQ(PadToWord<64>("01"), BitsToStream(bits));
  EXPECT_EQ(20, reader.ReadBits(&bits, 20));
  EXPECT_EQ(PadToWord<64>("01010111011101001100"), BitsToStream(bits));
  EXPECT_EQ(20, reader.ReadBits(&bits, 20));
  EXPECT_EQ(PadToWord<64>("11101110110000000000"), BitsToStream(bits));
  EXPECT_EQ(22, reader.ReadBits(&bits, 30));
  EXPECT_EQ(PadToWord<64>("0000000000000000000000"), BitsToStream(bits));
  EXPECT_TRUE(reader.ReachedEnd());
}

TEST(BitReaderWord64, FromU64) {
  std::vector<uint64_t> buffer = {
    0xAAAAAAAAAAAAAAAA,
    0xBBBBBBBBBBBBBBBB,
    0xCCCCCCCCCCCCCCCC,
    0xDDDDDDDDDDDDDDDD,
  };

  const std::string total_stream =
      "0101010101010101010101010101010101010101010101010101010101010101"
      "1101110111011101110111011101110111011101110111011101110111011101"
      "0011001100110011001100110011001100110011001100110011001100110011"
      "1011101110111011101110111011101110111011101110111011101110111011";

  BitReaderWord64 reader(std::move(buffer));

  uint64_t bits = 0;
  size_t pos = 0;
  size_t to_read = 5;
  while (reader.ReadBits(&bits, to_read) > 0) {
    EXPECT_EQ(BitsToStream(bits),
              PadToWord<64>(total_stream.substr(pos, to_read)));
    pos += to_read;
    to_read = (to_read + 35) % 64 + 1;
  }
  EXPECT_TRUE(reader.ReachedEnd());
}

TEST(BitReaderWord64, ComparisonLotsOfU8) {
  std::vector<uint8_t> buffer;
  for(uint32_t i = 0; i < 10003; ++i) {
    buffer.push_back(static_cast<uint8_t>(i % 255));
  }

  BitReaderFromString reader1(buffer);
  BitReaderWord64 reader2(std::move(buffer));

  uint64_t bits1 = 0, bits2 = 0;
  size_t to_read = 5;
  while (reader1.ReadBits(&bits1, to_read) > 0) {
    reader2.ReadBits(&bits2, to_read);
    EXPECT_EQ(bits1, bits2);
    to_read = (to_read + 35) % 64 + 1;
  }

  EXPECT_EQ(0, reader2.ReadBits(&bits2, 1));
}

TEST(BitReaderWord64, ComparisonLotsOfU64) {
  std::vector<uint64_t> buffer;
  for(uint64_t i = 0; i < 1000; ++i) {
    buffer.push_back(i);
  }

  BitReaderFromString reader1(buffer);
  BitReaderWord64 reader2(std::move(buffer));

  uint64_t bits1 = 0, bits2 = 0;
  size_t to_read = 5;
  while (reader1.ReadBits(&bits1, to_read) > 0) {
    reader2.ReadBits(&bits2, to_read);
    EXPECT_EQ(bits1, bits2);
    to_read = (to_read + 35) % 64 + 1;
  }

  EXPECT_EQ(0, reader2.ReadBits(&bits2, 1));
}

TEST(ReadWriteWord64, ReadWriteLotsOfBits) {
  BitWriterWord64 writer(16384);
  for (uint64_t i = 0; i < 65000; i += 25) {
    const uint64_t num_bits = i % 64 + 1;
    const uint64_t bits = i >> (64 - num_bits);
    writer.WriteBits(bits, num_bits);
  }

  BitReaderWord64 reader(writer.GetDataCopy());
  for (uint64_t i = 0; i < 65000; i += 25) {
    const uint64_t num_bits = i % 64 + 1;
    const uint64_t expected_bits = i >> (64 - num_bits);
    uint64_t bits = 0;
    reader.ReadBits(&bits, num_bits);
    EXPECT_EQ(expected_bits, bits);
  }

  EXPECT_TRUE(reader.OnlyZeroesLeft());
}

TEST(VariableWidthWrite, Write0U) {
  BitWriterStringStream writer;
  writer.WriteVariableWidthU64(0, 2);
  EXPECT_EQ("000", writer.GetStreamRaw ());
  writer.WriteVariableWidthU32(0, 2);
  EXPECT_EQ("000""000", writer.GetStreamRaw());
  writer.WriteVariableWidthU16(0, 2);
  EXPECT_EQ("000""000""000", writer.GetStreamRaw());
  writer.WriteVariableWidthU8(0, 2);
  EXPECT_EQ("000""000""000""000", writer.GetStreamRaw());
}

TEST(VariableWidthWrite, Write0S) {
  BitWriterStringStream writer;
  writer.WriteVariableWidthS64(0, 2);
  EXPECT_EQ("000", writer.GetStreamRaw ());
  writer.WriteVariableWidthS32(0, 2);
  EXPECT_EQ("000""000", writer.GetStreamRaw());
  writer.WriteVariableWidthS16(0, 2);
  EXPECT_EQ("000""000""000", writer.GetStreamRaw());
  writer.WriteVariableWidthS8(0, 2);
  EXPECT_EQ("000""000""000""000", writer.GetStreamRaw());
}

TEST(VariableWidthWrite, WriteSmallUnsigned) {
  BitWriterStringStream writer;
  writer.WriteVariableWidthU64(1, 2);
  EXPECT_EQ("100", writer.GetStreamRaw ());
  writer.WriteVariableWidthU32(2, 2);
  EXPECT_EQ("100""010", writer.GetStreamRaw());
  writer.WriteVariableWidthU16(3, 2);
  EXPECT_EQ("100""010""110", writer.GetStreamRaw());
  writer.WriteVariableWidthU8(4, 2);
  EXPECT_EQ("100""010""110""001100", writer.GetStreamRaw());
}

TEST(VariableWidthWrite, WriteSmallSigned) {
  BitWriterStringStream writer;
  writer.WriteVariableWidthS64(1, 2);
  EXPECT_EQ("010", writer.GetStreamRaw ());
  writer.WriteVariableWidthS64(-1, 2);
  EXPECT_EQ("010""100", writer.GetStreamRaw());
  EXPECT_EQ("010""100", writer.GetStreamRaw());
  writer.WriteVariableWidthS16(3, 2);
  EXPECT_EQ("010""100""011100", writer.GetStreamRaw());
  writer.WriteVariableWidthS8(-4, 2);
  EXPECT_EQ("010""100""011100""111100", writer.GetStreamRaw());
}

TEST(VariableWidthWrite, U64Val127ChunkLength7) {
  BitWriterStringStream writer;
  writer.WriteVariableWidthU64(127, 7);
  EXPECT_EQ("1111111""0", writer.GetStreamRaw());
}

TEST(VariableWidthWrite, U32Val255ChunkLength7) {
  BitWriterStringStream writer;
  writer.WriteVariableWidthU32(255, 7);
  EXPECT_EQ("1111111""1""1000000""0", writer.GetStreamRaw());
}

TEST(VariableWidthWrite, U16Val2ChunkLength4) {
  BitWriterStringStream writer;
  writer.WriteVariableWidthU16(2, 4);
  EXPECT_EQ("0100""0", writer.GetStreamRaw());
}

TEST(VariableWidthWrite, U8Val128ChunkLength7) {
  BitWriterStringStream writer;
  writer.WriteVariableWidthU8(128, 7);
  EXPECT_EQ("0000000""1""1", writer.GetStreamRaw());
}

TEST(VariableWidthWrite, U64ValAAAAChunkLength2) {
  BitWriterStringStream writer;
  writer.WriteVariableWidthU64(0xAAAA, 2);
  EXPECT_EQ("01""1""01""1""01""1""01""1"
            "01""1""01""1""01""1""01""0", writer.GetStreamRaw());
}

TEST(VariableWidthWrite, S8ValM128ChunkLength7) {
  BitWriterStringStream writer;
  writer.WriteVariableWidthS8(-128, 7);
  EXPECT_EQ("1111111""1""1", writer.GetStreamRaw());
}

TEST(VariableWidthRead, U64Val127ChunkLength7) {
  BitReaderFromString reader("1111111""0");
  uint64_t val = 0;
  ASSERT_TRUE(reader.ReadVariableWidthU64(&val, 7));
  EXPECT_EQ(127, val);
}

TEST(VariableWidthRead, U32Val255ChunkLength7) {
  BitReaderFromString reader("1111111""1""1000000""0");
  uint32_t val = 0;
  ASSERT_TRUE(reader.ReadVariableWidthU32(&val, 7));
  EXPECT_EQ(255, val);
}

TEST(VariableWidthRead, U16Val2ChunkLength4) {
  BitReaderFromString reader("0100""0");
  uint16_t val = 0;
  ASSERT_TRUE(reader.ReadVariableWidthU16(&val, 4));
  EXPECT_EQ(2, val);
}

TEST(VariableWidthRead, U8Val128ChunkLength7) {
  BitReaderFromString reader("0000000""1""1");
  uint8_t val = 0;
  ASSERT_TRUE(reader.ReadVariableWidthU8(&val, 7));
  EXPECT_EQ(128, val);
}

TEST(VariableWidthRead, U64ValAAAAChunkLength2) {
  BitReaderFromString reader("01""1""01""1""01""1""01""1"
                             "01""1""01""1""01""1""01""0");
  uint64_t val = 0;
  ASSERT_TRUE(reader.ReadVariableWidthU64(&val, 2));
  EXPECT_EQ(0xAAAA, val);
}

TEST(VariableWidthRead, S8ValM128ChunkLength7) {
  BitReaderFromString reader("1111111""1""1");
  int8_t val = 0;
  ASSERT_TRUE(reader.ReadVariableWidthS8(&val, 7));
  EXPECT_EQ(-128, val);
}

TEST(VariableWidthRead, FailTooShort) {
  BitReaderFromString reader("00000001100000");
  uint64_t val = 0;
  ASSERT_FALSE(reader.ReadVariableWidthU64(&val, 7));
}

TEST(VariableWidthWriteRead, SingleWriteReadU64) {
  for (uint64_t i = 0; i < 1000000; i += 1234) {
    const uint64_t val = i * i * i;
    const size_t chunk_length = i % 16 + 1;

    BitWriterWord64 writer;
    writer.WriteVariableWidthU64(val, chunk_length);

    BitReaderWord64 reader(writer.GetDataCopy());
    uint64_t read_val = 0;
    ASSERT_TRUE(reader.ReadVariableWidthU64(&read_val, chunk_length));

    EXPECT_EQ(val, read_val) << "Chunk length " << chunk_length;
  }
}

TEST(VariableWidthWriteRead, SingleWriteReadS64) {
  for (int64_t i = 0; i < 1000000; i += 4321) {
    const int64_t val = i * i * (i % 2 ? -i : i);
    const size_t chunk_length = i % 16 + 1;

    BitWriterWord64 writer;
    writer.WriteVariableWidthS64(val, chunk_length);

    BitReaderWord64 reader(writer.GetDataCopy());
    int64_t read_val = 0;
    ASSERT_TRUE(reader.ReadVariableWidthS64(&read_val, chunk_length));

    EXPECT_EQ(val, read_val) << "Chunk length " << chunk_length;
  }
}

TEST(VariableWidthWriteRead, SingleWriteReadU32) {
  for (uint32_t i = 0; i < 100000; i += 123) {
    const uint32_t val = i * i;
    const size_t chunk_length = i % 16 + 1;

    BitWriterWord64 writer;
    writer.WriteVariableWidthU32(val, chunk_length);

    BitReaderWord64 reader(writer.GetDataCopy());
    uint32_t read_val = 0;
    ASSERT_TRUE(reader.ReadVariableWidthU32(&read_val, chunk_length));

    EXPECT_EQ(val, read_val) << "Chunk length " << chunk_length;
  }
}

TEST(VariableWidthWriteRead, SingleWriteReadS32) {
  for (int32_t i = 0; i < 100000; i += 123) {
    const int32_t val = i * (i % 2 ? -i : i);
    const size_t chunk_length = i % 16 + 1;

    BitWriterWord64 writer;
    writer.WriteVariableWidthS32(val, chunk_length);

    BitReaderWord64 reader(writer.GetDataCopy());
    int32_t read_val = 0;
    ASSERT_TRUE(reader.ReadVariableWidthS32(&read_val, chunk_length));

    EXPECT_EQ(val, read_val) << "Chunk length " << chunk_length;
  }
}

TEST(VariableWidthWriteRead, SingleWriteReadU16) {
  for (int i = 0; i < 65536; i += 123) {
    const uint16_t val = static_cast<int16_t>(i);
    const size_t chunk_length = val % 10 + 1;

    BitWriterWord64 writer;
    writer.WriteVariableWidthU16(val, chunk_length);

    BitReaderWord64 reader(writer.GetDataCopy());
    uint16_t read_val = 0;
    ASSERT_TRUE(reader.ReadVariableWidthU16(&read_val, chunk_length));

    EXPECT_EQ(val, read_val) << "Chunk length " << chunk_length;
  }
}

TEST(VariableWidthWriteRead, SingleWriteReadS16) {
  for (int i = -32768; i < 32768; i += 123) {
    const int16_t val = static_cast<int16_t>(i);
    const size_t chunk_length = std::abs(i) % 10 + 1;


    BitWriterWord64 writer;
    writer.WriteVariableWidthS16(val, chunk_length);

    BitReaderWord64 reader(writer.GetDataCopy());
    int16_t read_val = 0;
    ASSERT_TRUE(reader.ReadVariableWidthS16(&read_val, chunk_length));

    EXPECT_EQ(val, read_val) << "Chunk length " << chunk_length;
  }
}

TEST(VariableWidthWriteRead, SingleWriteReadU8) {
  for (int i = 0; i < 256; ++i) {
    const uint8_t val = static_cast<uint8_t>(i);
    const size_t chunk_length = val % 5 + 1;

    BitWriterWord64 writer;
    writer.WriteVariableWidthU8(val, chunk_length);

    BitReaderWord64 reader(writer.GetDataCopy());
    uint8_t read_val = 0;
    ASSERT_TRUE(reader.ReadVariableWidthU8(&read_val, chunk_length));

    EXPECT_EQ(val, read_val) << "Chunk length " << chunk_length;
  }
}

TEST(VariableWidthWriteRead, SingleWriteReadS8) {
  for (int i = -128; i < 128; ++i) {
    const int8_t val = static_cast<int8_t>(i);
    const size_t chunk_length = std::abs(i) % 5 + 1;

    BitWriterWord64 writer;
    writer.WriteVariableWidthS8(val, chunk_length);

    BitReaderWord64 reader(writer.GetDataCopy());
    int8_t read_val = 0;
    ASSERT_TRUE(reader.ReadVariableWidthS8(&read_val, chunk_length));

    EXPECT_EQ(val, read_val) << "Chunk length " << chunk_length;
  }
}

TEST(VariableWidthWriteRead, SmallNumbersChunkLength4) {
  const std::vector<uint64_t> expected_values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  BitWriterWord64 writer;
  for (uint64_t val : expected_values) {
    writer.WriteVariableWidthU64(val, 4);
  }

  EXPECT_EQ(50, writer.GetNumBits());

  std::vector<uint64_t> actual_values;
  BitReaderWord64 reader(writer.GetDataCopy());
  while(!reader.OnlyZeroesLeft()) {
    uint64_t val = 0;
    ASSERT_TRUE(reader.ReadVariableWidthU64(&val, 4));
    actual_values.push_back(val);
  }

  EXPECT_EQ(expected_values, actual_values);
}

TEST(VariableWidthWriteRead, VariedNumbersChunkLength8) {
  const std::vector<uint64_t> expected_values = {1000, 0, 255, 4294967296};
  const size_t kExpectedNumBits = 9 * (2 + 1 + 1 + 5);

  BitWriterWord64 writer;
  for (uint64_t val : expected_values) {
    writer.WriteVariableWidthU64(val, 8);
  }

  EXPECT_EQ(kExpectedNumBits, writer.GetNumBits());

  std::vector<uint64_t> actual_values;
  BitReaderWord64 reader(writer.GetDataCopy());
  while (!reader.OnlyZeroesLeft()) {
    uint64_t val = 0;
    ASSERT_TRUE(reader.ReadVariableWidthU64(&val, 8));
    actual_values.push_back(val);
  }

  EXPECT_EQ(expected_values, actual_values);
}

}  // anonymous namespace
