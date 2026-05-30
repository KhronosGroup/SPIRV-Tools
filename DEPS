use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': 'fe126c727b925b7c3b2742f6f54690973216952b',

  'effcee_revision': 'ae38e040cbb7e83efa8bfbb4967e5b8c8c89b55a',

  'googletest_revision': '8736d2cd5c1dcba41170ed2fddca14021d4916c3',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '972a15cedd008d846f1a39b2e88ce48d7f166cbd',

  'spirv_headers_revision': '1e770e7de8373a8dd49f23416cf7ca4001d01040',

  'mimalloc_revision': 'fef6b0dd70f9d7fa0750b0d0b9fbb471203b94cd',
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

