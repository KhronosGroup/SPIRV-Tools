use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': 'ca81d343946a1775860791f77cde565b43f92cdd',

  'effcee_revision': 'aea1f4d62ca9ee2f44b5393e98e175e200a22e8e',

  'googletest_revision': '305e5a238b3c8d11266fbafd85520fb6b3184851',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': 'a771d3fbe7c432dc4db68360c6c0004fdde5646b',

  'spirv_headers_revision': '49a1fceb9b1d087f3c25ad5ec077bb0e46231297',
}

deps = {
  'external/abseil_cpp':
      Var('github') + '/abseil/abseil-cpp.git@' + Var('abseil_revision'),

  'external/effcee':
      Var('github') + '/google/effcee.git@' + Var('effcee_revision'),

  'external/googletest':
      Var('github') + '/google/googletest.git@' + Var('googletest_revision'),

  'external/protobuf':
      Var('github') + '/protocolbuffers/protobuf.git@' + Var('protobuf_revision'),

  'external/re2':
      Var('github') + '/google/re2.git@' + Var('re2_revision'),

  'external/spirv-headers':
      Var('github') +  '/KhronosGroup/SPIRV-Headers.git@' +
          Var('spirv_headers_revision'),
}

