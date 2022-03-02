load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "platforms",
    commit = "fbd0d188dac49fbcab3d2876a2113507e6fc68e9",
    shallow_since = "1644333305 -0500",
    remote = "https://github.com/bazelbuild/platforms.git",
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
