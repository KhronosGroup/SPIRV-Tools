"""Constants and macros for spirv-tools BUILD."""

COMMON_COPTS = [
    "-DSPIRV_CHECK_CONTEXT",
    "-DSPIRV_COLOR_TERMINAL",
] + select({
    "@platforms//os:windows": [],
    "//conditions:default": [
        "-DSPIRV_LINUX",
        "-DSPIRV_TIMER_ENABLED",
        "-fvisibility=hidden",
        "-fno-exceptions",
        "-fno-rtti",
        "-Wall",
        "-Wextra",
        "-Wnon-virtual-dtor",
        "-Wno-missing-field-initializers",
        "-Werror",
        "-Wno-long-long",
        "-Wshadow",
        "-Wundef",
        "-Wconversion",
        "-Wno-sign-conversion",
    ],
})

TEST_COPTS = COMMON_COPTS + [
] + select({
    "@platforms//os:windows": [
        # Disable C4503 "decorated name length exceeded" warning,
        # triggered by some heavily templated types.
        # We don't care much about that in test code.
        # Important to do since we have warnings-as-errors.
        "/wd4503",
    ],
    "//conditions:default": [
        "-Wno-undef",
        "-Wno-self-assign",
        "-Wno-shadow",
        "-Wno-unused-parameter",
    ],
})

def incompatible_with(incompatible_constraints):
    return select(_merge_dicts([{"//conditions:default": []}, {
        constraint: ["@platforms//:incompatible"]
        for constraint in incompatible_constraints
    }]))

SPIRV_CORE_GRAMMAR_JSON_FILE = "@spirv_headers//:spirv_core_grammar_unified1"
DEBUGINFO_GRAMMAR_JSON_FILE = "@spirv_headers//:spirv_ext_inst_debuginfo_grammar_unified1"
CLDEBUGINFO100_GRAMMAR_JSON_FILE = "@spirv_headers//:spirv_ext_inst_opencl_debuginfo_100_grammar_unified1"
SHDEBUGINFO100_GRAMMAR_JSON_FILE = "@spirv_headers//:spirv_ext_inst_nonsemantic_shader_debuginfo_100_grammar_unified1"

def _merge_dicts(dicts):
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged

# TODO(b/413743565): Remove after legacy grammars removed.
def generate_core_tables(version):
    if not version:
        fail("Must specify version", "version")

    grammars = dict(
        core_grammar = "@spirv_headers//:spirv_core_grammar_{}".format(version),
        debuginfo_grammar = DEBUGINFO_GRAMMAR_JSON_FILE,
        cldebuginfo_grammar = CLDEBUGINFO100_GRAMMAR_JSON_FILE,
    )

    outs = dict(
        core_insts_output = "core.insts-{}.inc".format(version),
        operand_kinds_output = "operand.kinds-{}.inc".format(version),
    )

    cmd = (
        "$(location :generate_grammar_tables)" +
        " --spirv-core-grammar=$(location {core_grammar})" +
        " --extinst-debuginfo-grammar=$(location {debuginfo_grammar})" +
        " --extinst-cldebuginfo100-grammar=$(location {cldebuginfo_grammar})" +
        " --core-insts-output=$(location {core_insts_output})" +
        " --operand-kinds-output=$(location {operand_kinds_output})"
    ).format(**_merge_dicts([grammars, outs]))

    native.genrule(
        name = "gen_core_tables_" + version,
        srcs = grammars.values(),
        outs = outs.values(),
        cmd = cmd,
        cmd_bat = cmd,
        tools = [":generate_grammar_tables"],
        visibility = ["//visibility:private"],
    )

def generate_compressed_tables():
    grammars = dict(
        core_grammar = SPIRV_CORE_GRAMMAR_JSON_FILE,
        debuginfo_grammar = DEBUGINFO_GRAMMAR_JSON_FILE,
        cldebuginfo_grammar = CLDEBUGINFO100_GRAMMAR_JSON_FILE,
    )

    outs = dict(
        core_tables_header_output = "core_tables_header.inc",
        core_tables_body_output = "core_tables_body.inc",
    )

    cmd = (
        "$(location :ggt)" +
        " --spirv-core-grammar=$(location {core_grammar})" +
        " --extinst-debuginfo-grammar=$(location {debuginfo_grammar})" +
        " --extinst-cldebuginfo100-grammar=$(location {cldebuginfo_grammar})" +
        " --core-tables-body-output=$(location {core_tables_body_output})" +
        " --core-tables-header-output=$(location {core_tables_header_output})"
    ).format(**_merge_dicts([grammars, outs]))

    native.genrule(
        name = "gen_compressed_tables",
        srcs = grammars.values(),
        outs = outs.values(),
        cmd = cmd,
        cmd_bat = cmd,
        tools = [":ggt"],
        visibility = ["//visibility:private"],
    )

def generate_vendor_tables(extension, target = "", operand_kind_prefix = ""):
    if not extension:
        fail("Must specify extension", "extension")

    if target == "":
        extension_rule = extension.replace("-", "_").replace(".", "_")
        grammars = dict(
            vendor_grammar = "@spirv_headers//:spirv_ext_inst_{}_grammar_unified1".format(extension_rule),
        )
    else:
        grammars = dict(
            vendor_grammar = "@spirv_headers//:{}".format(target),
        )
        extension_rule = target
    outs = dict(
        vendor_insts_output = "{}.insts.inc".format(extension),
    )
    cmd = (
        "$(location :generate_grammar_tables)" +
        " --extinst-vendor-grammar=$(location {vendor_grammar})" +
        " --vendor-insts-output=$(location {vendor_insts_output})" +
        " --vendor-operand-kind-prefix={operand_kind_prefix}"
    ).format(operand_kind_prefix = operand_kind_prefix, **_merge_dicts([grammars, outs]))

    native.genrule(
        name = "gen_vendor_tables_" + extension_rule,
        srcs = grammars.values(),
        outs = outs.values(),
        cmd = cmd,
        cmd_bat = cmd,
        tools = [":generate_grammar_tables"],
        visibility = ["//visibility:private"],
    )

def generate_extinst_lang_headers(name, grammar = None):
    if not grammar:
        fail("Must specify grammar", "grammar")
    outs = dict(
        extinst_output_path = name + ".h",
    )
    cmd = (
        "$(location :generate_language_headers)" +
        " --extinst-grammar=$<" +
        " --extinst-output-path=$(location {extinst_output_path})"
    ).format(**outs)

    native.genrule(
        name = "gen_extinst_lang_headers_{}".format(name),
        srcs = [grammar],
        outs = outs.values(),
        cmd = cmd,
        cmd_bat = cmd,
        tools = [":generate_language_headers"],
        visibility = ["//visibility:private"],
    )
