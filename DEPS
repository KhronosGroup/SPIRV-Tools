use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '41253e45ef6cc814ffe297db03a388379007f3cf',

  'effcee_revision': '910ed15722d5d05c9d71ecf36c1a22243cb79b02',

  'googletest_revision': '1fc11dea107bde2a92dad7a9ed1b960978f1065a',

  # Use a recent protobuf, which can depend on abseil
  'protobuf_revision': '35cd01f9fe9afbeea38cc7b979a3b6bfcde82c03',

  're2_revision': '972a15cedd008d846f1a39b2e88ce48d7f166cbd',

  'spirv_headers_revision': 'c63848ecf2200425511319fd8bf2c17b751e501e',

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

