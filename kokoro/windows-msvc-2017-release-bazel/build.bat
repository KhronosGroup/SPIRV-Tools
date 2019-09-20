:: Copyright (c) 2019 Google LLC.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
::
:: Windows Build Script.

@echo on

set BUILD_ROOT=%cd%
set SRC=%cd%\github\SPIRV-Tools
set BUILD_TYPE=%1
set VS_VERSION=%2

:: Force usage of python 3.6
set PATH=C:\python36;%PATH%

cd %SRC%
git clone --depth=1 https://github.com/KhronosGroup/SPIRV-Headers external/spirv-headers
git clone --depth=1 https://github.com/google/googletest          external/googletest
git clone --depth=1 https://github.com/google/effcee              external/effcee
git clone --depth=1 https://github.com/google/re2                 external/re2
git clone --depth=1 https://github.com/protocolbuffers/protobuf   external/protobuf

:: #########################################
:: set up msvc build env
:: #########################################
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
echo "Using VS 2017..."

cd %SRC%

:: #########################################
:: Start building.
:: #########################################
echo "Build everything... %DATE% %TIME%"

if "%KOKORO_GITHUB_COMMIT%." == "." (
  set BUILD_SHA=%KOKORO_GITHUB_PULL_REQUEST_COMMIT%
) else (
  set BUILD_SHA=%KOKORO_GITHUB_COMMIT%
)

bazel build :all

if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo "Build Completed %DATE% %TIME%"

:: ##############
:: Run the tests
:: ##############
echo "Running Tests... %DATE% %TIME%"
bazel test :all
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo "Tests Completed %DATE% %TIME%"

exit /b 0

