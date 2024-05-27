use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '1a31b81c0a467c1c8e229b9fc172a4eb0db5bd85',

  'effcee_revision': '19b4aa87af25cb4ee779a071409732f34bfc305c',

  'googletest_revision': '9b4993ca7d1279dec5c5d41ba327cb11a77bdc00',

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

