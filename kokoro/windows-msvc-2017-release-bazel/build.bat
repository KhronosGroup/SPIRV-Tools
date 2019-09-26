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

set SRC=%cd%\github\SPIRV-Tools

:: Force usage of python 3.6
set PATH=C:\python36;%PATH%

cd %SRC%

:: REM Fix up the MSYS environment.
:: wget -q http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-gcc-7.3.0-2-any.pkg.tar.xz
:: wget -q http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-gcc-libs-7.3.0-2-any.pkg.tar.xz
:: c:\tools\msys64\usr\bin\bash --login -c "pacman -R --noconfirm catgets libcatgets"
:: c:\tools\msys64\usr\bin\bash --login -c "pacman -Syu --noconfirm"
:: c:\tools\msys64\usr\bin\bash --login -c "pacman -Sy --noconfirm mingw-w64-x86_64-crt-git patch"
:: c:\tools\msys64\usr\bin\bash --login -c "pacman -U --noconfirm mingw-w64-x86_64-gcc*-7.3.0-2-any.pkg.tar.xz"
:: set PATH=c:\tools\msys64\mingw64\bin;c:\tools\msys64\usr\bin;%PATH%
:: set BAZEL_SH=C:\tools\msys64\usr\bin\bash.exe
:: REM Install Bazel.
:: wget -q https://github.com/bazelbuild/bazel/releases/download/0.29.1/bazel-0.29.1-windows-x86_64.zip
:: unzip -q bazel-0.29.1-windows-x86_64.zip

git clone --depth=1 https://github.com/KhronosGroup/SPIRV-Headers external/spirv-headers
git clone --depth=1 https://github.com/google/googletest          external/googletest
git clone --depth=1 https://github.com/google/effcee              external/effcee
git clone --depth=1 https://github.com/google/re2                 external/re2
git clone --depth=1 https://github.com/protocolbuffers/protobuf   external/protobuf

choco install bazel -y
:: %SRC%\bazel.exe --version

:: #########################################
:: Start building.
:: #########################################
echo "Build everything... %DATE% %TIME%"
bazel.exe build :all
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo "Build Completed %DATE% %TIME%"

:: ##############
:: Run the tests
:: ##############
echo "Running Tests... %DATE% %TIME%"
bazel.exe test :all
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo "Tests Completed %DATE% %TIME%"

exit /b 0

