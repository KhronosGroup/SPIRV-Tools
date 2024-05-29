use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '1a31b81c0a467c1c8e229b9fc172a4eb0db5bd85',

  'effcee_revision': '19b4aa87af25cb4ee779a071409732f34bfc305c',

  'googletest_revision': '9b4993ca7d1279dec5c5d41ba327cb11a77bdc00',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '917047f3606d3ba9e2de0d383c3cd80c94ed732c',

  'spirv_headers_revision': 'ea77f2a826bc820cb8f57f9b2a7c7eccb681c731',
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

