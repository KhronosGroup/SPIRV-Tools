load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    strip_prefix = "bazel-skylib-main",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/main.zip"],
)

local_repository(
    name = "spirv_headers",
    path = "external/spirv-headers",
)

local_repository(
    name = "com_google_googletest",
    path = "external/googletest",
)

local_repository(
    name = "com_googlesource_code_re2",
    path = "external/re2",
)

local_repository(
    name = "com_google_effcee",
    path = "external/effcee",
)

local_repository(
    name = "com_google_absl",
    path = "external/abseil_cpp",
)
