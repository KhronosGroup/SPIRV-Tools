use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '884282e28f6591d6c0680a79e07357a5aa868618',

  'effcee_revision': 'ae38e040cbb7e83efa8bfbb4967e5b8c8c89b55a',

  'googletest_revision': 'd72f9c8aea6817cdf1ca0ac10887f328de7f3da2',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '972a15cedd008d846f1a39b2e88ce48d7f166cbd',

  'spirv_headers_revision': '6dd7ba990830f7c15ac1345ff3b43ef6ffdad216',

  'mimalloc_revision': '75d69f4ab736ad9f56cdd76c7eb883f60ac48869',
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

  'external/mimalloc':
      Var('github') + '/microsoft/mimalloc.git@' + Var('mimalloc_revision'),
}

