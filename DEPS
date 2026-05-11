use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '30bba84041ba0aadd2c31b52742b8157db047a2f',

  'effcee_revision': '63394054b5afa14aee3e32bd12b227eb7225b871',

  'googletest_revision': 'd72f9c8aea6817cdf1ca0ac10887f328de7f3da2',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '972a15cedd008d846f1a39b2e88ce48d7f166cbd',

  'spirv_headers_revision': '58006c901d1d5c37dece6b6610e9af87fa951375',

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

