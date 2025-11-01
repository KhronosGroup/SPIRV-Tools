use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '6c5a175774752699ac20c5b1a329efd4710b2809',

  'effcee_revision': '514b52ec61609744d7e587d93a7ef9b60407ab45',

  'googletest_revision': '6ec14dfd8c409d05fba94e18e3a02df35b874353',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '61c4644171ee6b480540bf9e569cba06d9090b4b',

  'spirv_headers_revision': 'f2e4bd213104fe323a01e935df56557328d37ac8',

  'mimalloc_revision': '09a27098aa6e9286518bd9c74e6ffa7199c3f04e',
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

