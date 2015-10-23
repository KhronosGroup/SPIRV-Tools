// Copyright (c) 2015 The Khronos Group Inc.
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

#include "UnitSPIRV.h"
#include "util/hex_float.h"

#include <gmock/gmock.h>
#include <tuple>
#include <sstream>
#include <string>


namespace {
using ::testing::Eq;
using spvutils::BitwiseCast;

using HexFloatTest =
    ::testing::TestWithParam<std::pair<float, std::string>>;
using DecodeHexFloatTest =
    ::testing::TestWithParam<std::pair<std::string, float>>;

TEST_P(HexFloatTest, EncodeCorrectly) {
  std::stringstream ss;
  ss << spvutils::HexFloat<float>(std::get<0>(GetParam()));
  EXPECT_THAT(ss.str(), Eq(std::get<1>(GetParam())));
}

TEST_P(HexFloatTest, DecodeCorrectly) {
  std::stringstream ss(std::get<1>(GetParam()));
  spvutils::HexFloat<float> myFloat(0.f);
  ss >> myFloat;
  float expected = std::get<0>(GetParam());
  // Check that the two floats are bitwise equal. We do it this way because
  // nan != nan, and so the test would fail even if they were exactly the same.
  EXPECT_THAT(BitwiseCast<uint32_t>(myFloat.value()),
              Eq(BitwiseCast<uint32_t>(std::get<0>(GetParam()))));
}

INSTANTIATE_TEST_CASE_P(
    Float32Tests, HexFloatTest,
    ::testing::ValuesIn(std::vector<std::pair<float, std::string>>({
        {0.f, "0x0p+0"},
        {1.f, "0x1p+0"},
        {2.f, "0x1p+1"},
        {3.f, "0x1.8p+1"},
        {0.5f, "0x1p-1"},
        {0.25f, "0x1p-2"},
        {0.75f, "0x1.8p-1"},
        {-0.f, "-0x0p+0"},
        {-1.f, "-0x1p+0"},
        {-0.5f, "-0x1p-1"},
        {-0.25f, "-0x1p-2"},
        {-0.75f, "-0x1.8p-1"},

        // Larger numbers
        {512.f, "0x1p+9"},
        {-512.f, "-0x1p+9"},
        {1024.f, "0x1p+10"},
        {-1024.f, "-0x1p+10"},
        {1024.f + 8.f, "0x1.02p+10"},
        {-1024.f - 8.f, "-0x1.02p+10"},

        // Small numbers
        {1.0f / 512.f, "0x1p-9"},
        {1.0f / -512.f, "-0x1p-9"},
        {1.0f / 1024.f, "0x1p-10"},
        {1.0f / -1024.f, "-0x1p-10"},
        {1.0f / 1024.f + 1.0f / 8.f, "0x1.02p-3"},
        {1.0f / -1024.f - 1.0f / 8.f, "-0x1.02p-3"},

        // lowest non-denorm
        {ldexp(1.0f, -126), "0x1p-126"},
        {ldexp(-1.0f, -126), "-0x1p-126"},

        // Denormalized values
        {ldexp(1.0f, -127), "0x1p-127"},
        {ldexp(1.0f, -127) / 2.0f, "0x1p-128"},
        {ldexp(1.0f, -127) / 4.0f, "0x1p-129"},
        {ldexp(1.0f, -127) / 8.0f, "0x1p-130"},
        {ldexp(-1.0f, -127), "-0x1p-127"},
        {ldexp(-1.0f, -127) / 2.0f, "-0x1p-128"},
        {ldexp(-1.0f, -127) / 4.0f, "-0x1p-129"},
        {ldexp(-1.0f, -127) / 8.0f, "-0x1p-130"},

        {ldexp(1.0, -127) + (ldexp(1.0, -127) / 2.0f), "0x1.8p-127"},
        {ldexp(1.0, -127) / 2.0 + (ldexp(1.0, -127) / 4.0f), "0x1.8p-128"},

        // Various NAN and INF cases
        {BitwiseCast<float>(0xFF800000), "-0x1p+128"},         // -inf
        {BitwiseCast<float>(0x7F800000), "0x1p+128"},          // inf
        {BitwiseCast<float>(0xFFC00000), "-0x1.8p+128"},       // -nan
        {BitwiseCast<float>(0xFF800100), "-0x1.0002p+128"},    // -nan
        {BitwiseCast<float>(0xFF800c00), "-0x1.0018p+128"},    // -nan
        {BitwiseCast<float>(0xFF80F000), "-0x1.01ep+128"},     // -nan
        {BitwiseCast<float>(0xFFFFFFFF), "-0x1.fffffep+128"},  // -nan
        {BitwiseCast<float>(0x7FC00000), "0x1.8p+128"},        // +nan
        {BitwiseCast<float>(0x7F800100), "0x1.0002p+128"},     // +nan
        {BitwiseCast<float>(0x7f800c00), "0x1.0018p+128"},     // +nan
        {BitwiseCast<float>(0x7F80F000), "0x1.01ep+128"},      // +nan
        {BitwiseCast<float>(0x7FFFFFFF), "0x1.fffffep+128"},   // +nan
    })));

TEST_P(DecodeHexFloatTest, DecodeCorrectly) {
  std::stringstream ss(std::get<0>(GetParam()));
  spvutils::HexFloat<float> myFloat(0.f);
  ss >> myFloat;
  float expected = std::get<1>(GetParam());
  // Check that the two floats are bitwise equal. We do it this way because
  // nan != nan, and so the test would fail even if they were exactly the same.
  EXPECT_THAT(BitwiseCast<uint32_t>(myFloat.value()),
              Eq(BitwiseCast<uint32_t>(std::get<1>(GetParam()))));
}

INSTANTIATE_TEST_CASE_P(
    Float32DecodeTests, DecodeHexFloatTest,
    ::testing::ValuesIn(std::vector<std::pair<std::string, float>>({
        {"0x0p+000", 0.f},
        {"0x0p0", 0.f},
        {"0x0p-0", 0.f},

        // inf cases
        {"-0x1p+128", BitwiseCast<float>(0xFF800000)},         // -inf
        {"0x32p+127", BitwiseCast<float>(0x7F800000)},         // inf
        {"0x32p+500", BitwiseCast<float>(0x7F800000)},         // inf
        {"-0x32p+127", BitwiseCast<float>(0xFF800000)},         // -inf

        // flush to zero cases
        {"0x1p-500", 0.f}, // Exponent underflows.
        {"-0x1p-500", -0.f},
        {"0x0.00000000001p-126", 0.f}, // Fraction causes underfloat.
        {"-0x0.0000000001p-127", -0.f},
        {"-0x0.01p-142", -0.f}, // Fraction causes undeflow to underflow more.
        {"0x0.01p-142", 0.f},

        // Some floats that do not encode the same way as the decode.
        {"0x2p+0", 2.f},
        {"0xFFp+0", 255.f},
        {"0x0.8p+0", 0.5f},
        {"0x0.4p+0", 0.25f},
    })));
// TODO(awoloszyn): Add some more decoding tests with "abnormal" formats.
// TODO(awoloszyn): Add double tests
// TODO(awoloszyn): Add fp16 tests and HexFloatTraits.

}
