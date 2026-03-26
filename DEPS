use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '04f3bc01d12cf58c90a1bb68990f087fa3c3ed19',

  'effcee_revision': 'ae38e040cbb7e83efa8bfbb4967e5b8c8c89b55a',

  'googletest_revision': '015950a936fc6b7cadca20eb0dbf6f054e6637fb',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '972a15cedd008d846f1a39b2e88ce48d7f166cbd',

  'spirv_headers_revision': 'f88a2d766840fc825af1fc065977953ba1fa4a91',

  'mimalloc_revision': '8ff03b636192e25db17eaaff29e6f75acc9a662b',
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

