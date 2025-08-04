export ANDROID_NDK=$HOME/Android/Sdk/ndk/27.0.12077973

mkdir -p build && cd build
mkdir -p libs
mkdir -p app

$ANDROID_NDK/ndk-build -j8 -C ../android_test     \
                      NDK_PROJECT_PATH=.      \
                      NDK_LIBS_OUT=`pwd`/libs \
                      NDK_APP_OUT=`pwd`/app
