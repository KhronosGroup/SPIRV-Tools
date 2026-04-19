use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': 'b85d16902fc47ac4e71efd2f44bdcf87ee65cef2',

  'effcee_revision': 'ae38e040cbb7e83efa8bfbb4967e5b8c8c89b55a',

  'googletest_revision': 'd72f9c8aea6817cdf1ca0ac10887f328de7f3da2',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '972a15cedd008d846f1a39b2e88ce48d7f166cbd',

  'spirv_headers_revision': 'ad9184e76a66b1001c29db9b0a3e87f646c64de0',

  'mimalloc_revision': 'ef1d67e51d98ceba1eefc5b4cd65255cbd5b7eff',
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

