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

#ifdef _MSC_VER
// We define this so that we can use sscanf in the tests without
// MSVC warning us all the time.
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <tuple>

#include <gmock/gmock.h>
#include "UnitSPIRV.h"
#include "util/hex_float.h"

namespace {
using ::testing::Eq;
using spvutils::BitwiseCast;
using spvutils::FloatProxy;

// In this file "encode" means converting a number into a string,
// and "decode" means converting a string into a number.

using HexFloatTest =
    ::testing::TestWithParam<std::pair<FloatProxy<float>, std::string>>;
using DecodeHexFloatTest =
    ::testing::TestWithParam<std::pair<std::string, FloatProxy<float>>>;
using HexDoubleTest =
    ::testing::TestWithParam<std::pair<FloatProxy<double>, std::string>>;
using DecodeHexDoubleTest =
    ::testing::TestWithParam<std::pair<std::string, FloatProxy<double>>>;

// Hex-encodes a float value.
template <typename T>
std::string EncodeViaHexFloat(const T& value) {
  std::stringstream ss;
  ss << spvutils::HexFloat<T>(value);
  return ss.str();
}

// The following two tests can't be DRY because they take different parameter
// types.

TEST_P(HexFloatTest, EncodeCorrectly) {
  EXPECT_THAT(EncodeViaHexFloat(GetParam().first), Eq(GetParam().second));
}

TEST_P(HexDoubleTest, EncodeCorrectly) {
  EXPECT_THAT(EncodeViaHexFloat(GetParam().first), Eq(GetParam().second));
}

// Decodes a hex-float string.
template <typename T>
FloatProxy<T> Decode(const std::string& str) {
  spvutils::HexFloat<FloatProxy<T>> decoded(0.f);
  EXPECT_TRUE((std::stringstream(str) >> decoded).eof());
  return decoded.value();
}

TEST_P(HexFloatTest, DecodeCorrectly) {
  EXPECT_THAT(Decode<float>(GetParam().second), Eq(GetParam().first));
}

TEST_P(HexDoubleTest, DecodeCorrectly) {
  EXPECT_THAT(Decode<double>(GetParam().second), Eq(GetParam().first));
}

INSTANTIATE_TEST_CASE_P(
    Float32Tests, HexFloatTest,
    ::testing::ValuesIn(std::vector<std::pair<FloatProxy<float>, std::string>>({
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
        {float(ldexp(1.0f, -126)), "0x1p-126"},
        {float(ldexp(-1.0f, -126)), "-0x1p-126"},

        // Denormalized values
        {float(ldexp(1.0f, -127)), "0x1p-127"},
        {float(ldexp(1.0f, -127) / 2.0f), "0x1p-128"},
        {float(ldexp(1.0f, -127) / 4.0f), "0x1p-129"},
        {float(ldexp(1.0f, -127) / 8.0f), "0x1p-130"},
        {float(ldexp(-1.0f, -127)), "-0x1p-127"},
        {float(ldexp(-1.0f, -127) / 2.0f), "-0x1p-128"},
        {float(ldexp(-1.0f, -127) / 4.0f), "-0x1p-129"},
        {float(ldexp(-1.0f, -127) / 8.0f), "-0x1p-130"},

        {float(ldexp(1.0, -127) + (ldexp(1.0, -127) / 2.0f)), "0x1.8p-127"},
        {float(ldexp(1.0, -127) / 2.0 + (ldexp(1.0, -127) / 4.0f)),
         "0x1.8p-128"},

    })));

INSTANTIATE_TEST_CASE_P(
    Float32NanTests, HexFloatTest,
    ::testing::ValuesIn(std::vector<std::pair<FloatProxy<float>, std::string>>({
        // Various NAN and INF cases
        {uint32_t(0xFF800000), "-0x1p+128"},         // -inf
        {uint32_t(0x7F800000), "0x1p+128"},          // inf
        {uint32_t(0xFFC00000), "-0x1.8p+128"},       // -nan
        {uint32_t(0xFF800100), "-0x1.0002p+128"},    // -nan
        {uint32_t(0xFF800c00), "-0x1.0018p+128"},    // -nan
        {uint32_t(0xFF80F000), "-0x1.01ep+128"},     // -nan
        {uint32_t(0xFFFFFFFF), "-0x1.fffffep+128"},  // -nan
        {uint32_t(0x7FC00000), "0x1.8p+128"},        // +nan
        {uint32_t(0x7F800100), "0x1.0002p+128"},     // +nan
        {uint32_t(0x7f800c00), "0x1.0018p+128"},     // +nan
        {uint32_t(0x7F80F000), "0x1.01ep+128"},      // +nan
        {uint32_t(0x7FFFFFFF), "0x1.fffffep+128"},   // +nan
    })));

INSTANTIATE_TEST_CASE_P(
    Float64Tests, HexDoubleTest,
    ::testing::ValuesIn(
        std::vector<std::pair<FloatProxy<double>, std::string>>({
            {0., "0x0p+0"},
            {1., "0x1p+0"},
            {2., "0x1p+1"},
            {3., "0x1.8p+1"},
            {0.5, "0x1p-1"},
            {0.25, "0x1p-2"},
            {0.75, "0x1.8p-1"},
            {-0., "-0x0p+0"},
            {-1., "-0x1p+0"},
            {-0.5, "-0x1p-1"},
            {-0.25, "-0x1p-2"},
            {-0.75, "-0x1.8p-1"},

            // Larger numbers
            {512., "0x1p+9"},
            {-512., "-0x1p+9"},
            {1024., "0x1p+10"},
            {-1024., "-0x1p+10"},
            {1024. + 8., "0x1.02p+10"},
            {-1024. - 8., "-0x1.02p+10"},

            // Large outside the range of normal floats
            {ldexp(1.0, 128), "0x1p+128"},
            {ldexp(1.0, 129), "0x1p+129"},
            {ldexp(-1.0, 128), "-0x1p+128"},
            {ldexp(-1.0, 129), "-0x1p+129"},
            {ldexp(1.0, 128) + ldexp(1.0, 90), "0x1.0000000004p+128"},
            {ldexp(1.0, 129) + ldexp(1.0, 120), "0x1.008p+129"},
            {ldexp(-1.0, 128) + ldexp(1.0, 90), "-0x1.fffffffff8p+127"},
            {ldexp(-1.0, 129) + ldexp(1.0, 120), "-0x1.ffp+128"},

            // Small numbers
            {1.0 / 512., "0x1p-9"},
            {1.0 / -512., "-0x1p-9"},
            {1.0 / 1024., "0x1p-10"},
            {1.0 / -1024., "-0x1p-10"},
            {1.0 / 1024. + 1.0 / 8., "0x1.02p-3"},
            {1.0 / -1024. - 1.0 / 8., "-0x1.02p-3"},

            // Small outside the range of normal floats
            {ldexp(1.0, -128), "0x1p-128"},
            {ldexp(1.0, -129), "0x1p-129"},
            {ldexp(-1.0, -128), "-0x1p-128"},
            {ldexp(-1.0, -129), "-0x1p-129"},
            {ldexp(1.0, -128) + ldexp(1.0, -90), "0x1.0000000004p-90"},
            {ldexp(1.0, -129) + ldexp(1.0, -120), "0x1.008p-120"},
            {ldexp(-1.0, -128) + ldexp(1.0, -90), "0x1.fffffffff8p-91"},
            {ldexp(-1.0, -129) + ldexp(1.0, -120), "0x1.ffp-121"},

            // lowest non-denorm
            {ldexp(1.0, -1022), "0x1p-1022"},
            {ldexp(-1.0, -1022), "-0x1p-1022"},

            // Denormalized values
            {ldexp(1.0, -1023), "0x1p-1023"},
            {ldexp(1.0, -1023) / 2.0, "0x1p-1024"},
            {ldexp(1.0, -1023) / 4.0, "0x1p-1025"},
            {ldexp(1.0, -1023) / 8.0, "0x1p-1026"},
            {ldexp(-1.0, -1024), "-0x1p-1024"},
            {ldexp(-1.0, -1024) / 2.0, "-0x1p-1025"},
            {ldexp(-1.0, -1024) / 4.0, "-0x1p-1026"},
            {ldexp(-1.0, -1024) / 8.0, "-0x1p-1027"},

            {ldexp(1.0, -1023) + (ldexp(1.0, -1023) / 2.0), "0x1.8p-1023"},
            {ldexp(1.0, -1023) / 2.0 + (ldexp(1.0, -1023) / 4.0),
             "0x1.8p-1024"},

        })));

INSTANTIATE_TEST_CASE_P(
    Float64NanTests, HexDoubleTest,
    ::testing::ValuesIn(std::vector<
                        std::pair<FloatProxy<double>, std::string>>({
        // Various NAN and INF cases
        {uint64_t(0xFFF0000000000000LL), "-0x1p+1024"},                //-inf
        {uint64_t(0x7FF0000000000000LL), "0x1p+1024"},                 //+inf
        {uint64_t(0xFFF8000000000000LL), "-0x1.8p+1024"},              // -nan
        {uint64_t(0xFFF0F00000000000LL), "-0x1.0fp+1024"},             // -nan
        {uint64_t(0xFFF0000000000001LL), "-0x1.0000000000001p+1024"},  // -nan
        {uint64_t(0xFFF0000300000000LL), "-0x1.00003p+1024"},          // -nan
        {uint64_t(0xFFFFFFFFFFFFFFFFLL), "-0x1.fffffffffffffp+1024"},  // -nan
        {uint64_t(0x7FF8000000000000LL), "0x1.8p+1024"},               // +nan
        {uint64_t(0x7FF0F00000000000LL), "0x1.0fp+1024"},              // +nan
        {uint64_t(0x7FF0000000000001LL), "0x1.0000000000001p+1024"},   // -nan
        {uint64_t(0x7FF0000300000000LL), "0x1.00003p+1024"},           // -nan
        {uint64_t(0x7FFFFFFFFFFFFFFFLL), "0x1.fffffffffffffp+1024"},   // -nan
    })));

TEST_P(DecodeHexFloatTest, DecodeCorrectly) {
  EXPECT_THAT(Decode<float>(GetParam().first), Eq(GetParam().second));
}

TEST_P(DecodeHexDoubleTest, DecodeCorrectly) {
  EXPECT_THAT(Decode<double>(GetParam().first), Eq(GetParam().second));
}

INSTANTIATE_TEST_CASE_P(
    Float32DecodeTests, DecodeHexFloatTest,
    ::testing::ValuesIn(std::vector<std::pair<std::string, FloatProxy<float>>>({
        {"0x0p+000", 0.f},
        {"0x0p0", 0.f},
        {"0x0p-0", 0.f},

        // flush to zero cases
        {"0x1p-500", 0.f},  // Exponent underflows.
        {"-0x1p-500", -0.f},
        {"0x0.00000000001p-126", 0.f},  // Fraction causes underflow.
        {"-0x0.0000000001p-127", -0.f},
        {"-0x0.01p-142", -0.f},  // Fraction causes additional underflow.
        {"0x0.01p-142", 0.f},

        // Some floats that do not encode the same way as they decode.
        {"0x2p+0", 2.f},
        {"0xFFp+0", 255.f},
        {"0x0.8p+0", 0.5f},
        {"0x0.4p+0", 0.25f},
    })));

INSTANTIATE_TEST_CASE_P(
    Float32DecodeInfTests, DecodeHexFloatTest,
    ::testing::ValuesIn(std::vector<std::pair<std::string, FloatProxy<float>>>({
        // inf cases
        {"-0x1p+128", uint32_t(0xFF800000)},   // -inf
        {"0x32p+127", uint32_t(0x7F800000)},   // inf
        {"0x32p+500", uint32_t(0x7F800000)},   // inf
        {"-0x32p+127", uint32_t(0xFF800000)},  // -inf
    })));

INSTANTIATE_TEST_CASE_P(
    Float64DecodeTests, DecodeHexDoubleTest,
    ::testing::ValuesIn(
        std::vector<std::pair<std::string, FloatProxy<double>>>({
            {"0x0p+000", 0.},
            {"0x0p0", 0.},
            {"0x0p-0", 0.},

            // flush to zero cases
            {"0x1p-5000", 0.},  // Exponent underflows.
            {"-0x1p-5000", -0.},
            {"0x0.0000000000000001p-1023", 0.},  // Fraction causes underflow.
            {"-0x0.000000000000001p-1024", -0.},
            {"-0x0.01p-1090", -0.f},  // Fraction causes additional underflow.
            {"0x0.01p-1090", 0.},

            // Some floats that do not encode the same way as they decode.
            {"0x2p+0", 2.},
            {"0xFFp+0", 255.},
            {"0x0.8p+0", 0.5},
            {"0x0.4p+0", 0.25},
        })));

INSTANTIATE_TEST_CASE_P(
    Float64DecodeInfTests, DecodeHexDoubleTest,
    ::testing::ValuesIn(
        std::vector<std::pair<std::string, FloatProxy<double>>>({
            // inf cases
            {"-0x1p+1024", uint64_t(0xFFF0000000000000)},   // -inf
            {"0x32p+1023", uint64_t(0x7FF0000000000000)},   // inf
            {"0x32p+5000", uint64_t(0x7FF0000000000000)},   // inf
            {"-0x32p+1023", uint64_t(0xFFF0000000000000)},  // -inf
        })));

TEST(FloatProxy, ValidConversion) {
  EXPECT_THAT(FloatProxy<float>(1.f).getAsFloat(), Eq(1.0f));
  EXPECT_THAT(FloatProxy<float>(32.f).getAsFloat(), Eq(32.0f));
  EXPECT_THAT(FloatProxy<float>(-1.f).getAsFloat(), Eq(-1.0f));
  EXPECT_THAT(FloatProxy<float>(0.f).getAsFloat(), Eq(0.0f));
  EXPECT_THAT(FloatProxy<float>(-0.f).getAsFloat(), Eq(-0.0f));
  EXPECT_THAT(FloatProxy<float>(1.2e32f).getAsFloat(), Eq(1.2e32f));

  EXPECT_TRUE(std::isinf(FloatProxy<float>(uint32_t(0xFF800000)).getAsFloat()));
  EXPECT_TRUE(std::isinf(FloatProxy<float>(uint32_t(0x7F800000)).getAsFloat()));
  EXPECT_TRUE(std::isnan(FloatProxy<float>(uint32_t(0xFFC00000)).getAsFloat()));
  EXPECT_TRUE(std::isnan(FloatProxy<float>(uint32_t(0xFF800100)).getAsFloat()));
  EXPECT_TRUE(std::isnan(FloatProxy<float>(uint32_t(0xFF800c00)).getAsFloat()));
  EXPECT_TRUE(std::isnan(FloatProxy<float>(uint32_t(0xFF80F000)).getAsFloat()));
  EXPECT_TRUE(std::isnan(FloatProxy<float>(uint32_t(0xFFFFFFFF)).getAsFloat()));
  EXPECT_TRUE(std::isnan(FloatProxy<float>(uint32_t(0x7FC00000)).getAsFloat()));
  EXPECT_TRUE(std::isnan(FloatProxy<float>(uint32_t(0x7F800100)).getAsFloat()));
  EXPECT_TRUE(std::isnan(FloatProxy<float>(uint32_t(0x7f800c00)).getAsFloat()));
  EXPECT_TRUE(std::isnan(FloatProxy<float>(uint32_t(0x7F80F000)).getAsFloat()));
  EXPECT_TRUE(std::isnan(FloatProxy<float>(uint32_t(0x7FFFFFFF)).getAsFloat()));

  EXPECT_THAT(FloatProxy<float>(uint32_t(0xFF800000)).data(), Eq(0xFF800000));
  EXPECT_THAT(FloatProxy<float>(uint32_t(0x7F800000)).data(), Eq(0x7F800000));
  EXPECT_THAT(FloatProxy<float>(uint32_t(0xFFC00000)).data(), Eq(0xFFC00000));
  EXPECT_THAT(FloatProxy<float>(uint32_t(0xFF800100)).data(), Eq(0xFF800100));
  EXPECT_THAT(FloatProxy<float>(uint32_t(0xFF800c00)).data(), Eq(0xFF800c00));
  EXPECT_THAT(FloatProxy<float>(uint32_t(0xFF80F000)).data(), Eq(0xFF80F000));
  EXPECT_THAT(FloatProxy<float>(uint32_t(0xFFFFFFFF)).data(), Eq(0xFFFFFFFF));
  EXPECT_THAT(FloatProxy<float>(uint32_t(0x7FC00000)).data(), Eq(0x7FC00000));
  EXPECT_THAT(FloatProxy<float>(uint32_t(0x7F800100)).data(), Eq(0x7F800100));
  EXPECT_THAT(FloatProxy<float>(uint32_t(0x7f800c00)).data(), Eq(0x7f800c00));
  EXPECT_THAT(FloatProxy<float>(uint32_t(0x7F80F000)).data(), Eq(0x7F80F000));
  EXPECT_THAT(FloatProxy<float>(uint32_t(0x7FFFFFFF)).data(), Eq(0x7FFFFFFF));
}

TEST(FloatProxy, Nan) {
  EXPECT_TRUE(FloatProxy<float>(uint32_t(0xFFC00000)).isNan());
  EXPECT_TRUE(FloatProxy<float>(uint32_t(0xFF800100)).isNan());
  EXPECT_TRUE(FloatProxy<float>(uint32_t(0xFF800c00)).isNan());
  EXPECT_TRUE(FloatProxy<float>(uint32_t(0xFF80F000)).isNan());
  EXPECT_TRUE(FloatProxy<float>(uint32_t(0xFFFFFFFF)).isNan());
  EXPECT_TRUE(FloatProxy<float>(uint32_t(0x7FC00000)).isNan());
  EXPECT_TRUE(FloatProxy<float>(uint32_t(0x7F800100)).isNan());
  EXPECT_TRUE(FloatProxy<float>(uint32_t(0x7f800c00)).isNan());
  EXPECT_TRUE(FloatProxy<float>(uint32_t(0x7F80F000)).isNan());
  EXPECT_TRUE(FloatProxy<float>(uint32_t(0x7FFFFFFF)).isNan());
}

TEST(FloatProxy, Negation) {
  EXPECT_THAT((-FloatProxy<float>(1.f)).getAsFloat(), Eq(-1.0f));
  EXPECT_THAT((-FloatProxy<float>(0.f)).getAsFloat(), Eq(-0.0f));

  EXPECT_THAT((-FloatProxy<float>(-1.f)).getAsFloat(), Eq(1.0f));
  EXPECT_THAT((-FloatProxy<float>(-0.f)).getAsFloat(), Eq(0.0f));

  EXPECT_THAT((-FloatProxy<float>(32.f)).getAsFloat(), Eq(-32.0f));
  EXPECT_THAT((-FloatProxy<float>(-32.f)).getAsFloat(), Eq(32.0f));

  EXPECT_THAT((-FloatProxy<float>(1.2e32f)).getAsFloat(), Eq(-1.2e32f));
  EXPECT_THAT((-FloatProxy<float>(-1.2e32f)).getAsFloat(), Eq(1.2e32f));

  EXPECT_THAT(
      (-FloatProxy<float>(std::numeric_limits<float>::infinity())).getAsFloat(),
      Eq(-std::numeric_limits<float>::infinity()));
  EXPECT_THAT((-FloatProxy<float>(-std::numeric_limits<float>::infinity()))
                  .getAsFloat(),
              Eq(std::numeric_limits<float>::infinity()));
}

// Test conversion of FloatProxy values to strings.
//
// In previous cases, we always wrapped the FloatProxy value in a HexFloat
// before conversion to a string.  In the following cases, the FloatProxy
// decides for itself whether to print as a regular number or as a hex float.

using FloatProxyFloatTest =
    ::testing::TestWithParam<std::pair<FloatProxy<float>, std::string>>;
using FloatProxyDoubleTest =
    ::testing::TestWithParam<std::pair<FloatProxy<double>, std::string>>;

// Converts a float value to a string via a FloatProxy.
template <typename T>
std::string EncodeViaFloatProxy(const T& value) {
  std::stringstream ss;
  ss << value;
  return ss.str();
}

// Converts a floating point string so that the exponent prefix
// is 'e', and the exponent value does not have leading zeros.
// The Microsoft runtime library likes to write things like "2.5E+010".
// Convert that to "2.5e+10".
// We don't care what happens to strings that are not floating point
// strings.
std::string NormalizeExponentInFloatString(std::string in) {
  std::string result;
  // Reserve one spot for the terminating null, even when the sscanf fails.
  std::vector<char> prefix(in.size()+1);
  char e;
  char plus_or_minus;
  int exponent;  // in base 10
  if ((4 == std::sscanf(in.c_str(), "%[-+.0123456789]%c%c%d", prefix.data(), &e,
                        &plus_or_minus, &exponent)) &&
      (e == 'e' || e == 'E') &&
      (plus_or_minus == '-' || plus_or_minus == '+')) {
    // It looks like a floating point value with exponent.
    std::stringstream out;
    out << prefix.data() << 'e' << plus_or_minus << exponent;
    result = out.str();
  } else {
    result = in;
  }
  return result;
}

TEST(NormalizeFloat, Sample) {
  EXPECT_THAT(NormalizeExponentInFloatString(""), Eq(""));
  EXPECT_THAT(NormalizeExponentInFloatString("1e-12"), Eq("1e-12"));
  EXPECT_THAT(NormalizeExponentInFloatString("1E+14"), Eq("1e+14"));
  EXPECT_THAT(NormalizeExponentInFloatString("1e-0012"), Eq("1e-12"));
  EXPECT_THAT(NormalizeExponentInFloatString("1.263E+014"), Eq("1.263e+14"));
}

// The following two tests can't be DRY because they take different parameter
// types.
TEST_P(FloatProxyFloatTest, EncodeCorrectly) {
  EXPECT_THAT(
      NormalizeExponentInFloatString(EncodeViaFloatProxy(GetParam().first)),
      Eq(GetParam().second));
}

TEST_P(FloatProxyDoubleTest, EncodeCorrectly) {
  EXPECT_THAT(
      NormalizeExponentInFloatString(EncodeViaFloatProxy(GetParam().first)),
      Eq(GetParam().second));
}

INSTANTIATE_TEST_CASE_P(
    Float32Tests, FloatProxyFloatTest,
    ::testing::ValuesIn(std::vector<std::pair<FloatProxy<float>, std::string>>({
        // Zero
        {0.f, "0"},
        // Normal numbers
        {1.f, "1"},
        {-0.25f, "-0.25"},
        {1000.0f, "1000"},

        // Still normal numbers, but with large magnitude exponents.
        {float(ldexp(1.f, 126)), "8.50706e+37"},
        {float(ldexp(-1.f, -126)), "-1.17549e-38"},

        // denormalized values are printed as hex floats.
        {float(ldexp(1.0f, -127)), "0x1p-127"},
        {float(ldexp(1.5f, -128)), "0x1.8p-128"},
        {float(ldexp(1.25, -129)), "0x1.4p-129"},
        {float(ldexp(1.125, -130)), "0x1.2p-130"},
        {float(ldexp(-1.0f, -127)), "-0x1p-127"},
        {float(ldexp(-1.0f, -128)), "-0x1p-128"},
        {float(ldexp(-1.0f, -129)), "-0x1p-129"},
        {float(ldexp(-1.5f, -130)), "-0x1.8p-130"},

        // NaNs
        {FloatProxy<float>(uint32_t(0xFFC00000)), "-0x1.8p+128"},
        {FloatProxy<float>(uint32_t(0xFF800100)), "-0x1.0002p+128"},

        {std::numeric_limits<float>::infinity(), "0x1p+128"},
        {-std::numeric_limits<float>::infinity(), "-0x1p+128"},
    })));

INSTANTIATE_TEST_CASE_P(
    Float64Tests, FloatProxyDoubleTest,
    ::testing::ValuesIn(
        std::vector<std::pair<FloatProxy<double>, std::string>>({
            {0., "0"},
            {1., "1"},
            {-0.25, "-0.25"},
            {1000.0, "1000"},

            // Large outside the range of normal floats
            {ldexp(1.0, 128), "3.40282366920938e+38"},
            {ldexp(1.5, 129), "1.02084710076282e+39"},
            {ldexp(-1.0, 128), "-3.40282366920938e+38"},
            {ldexp(-1.5, 129), "-1.02084710076282e+39"},

            // Small outside the range of normal floats
            {ldexp(1.5, -129), "2.20405190779179e-39"},
            {ldexp(-1.5, -129), "-2.20405190779179e-39"},

            // lowest non-denorm
            {ldexp(1.0, -1022), "2.2250738585072e-308"},
            {ldexp(-1.0, -1022), "-2.2250738585072e-308"},

            // Denormalized values
            {ldexp(1.125, -1023), "0x1.2p-1023"},
            {ldexp(-1.375, -1024), "-0x1.6p-1024"},

            // NaNs
            {uint64_t(0x7FF8000000000000LL), "0x1.8p+1024"},
            {uint64_t(0xFFF0F00000000000LL), "-0x1.0fp+1024"},

            // Infinity
            {std::numeric_limits<double>::infinity(), "0x1p+1024"},
            {-std::numeric_limits<double>::infinity(), "-0x1p+1024"},

        })));

// TODO(awoloszyn): Add fp16 tests and HexFloatTraits.
}
