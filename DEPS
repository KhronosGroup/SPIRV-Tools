use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '860dcf1c0c1714ac1dd940a25f653ea1eee954d7',

  'effcee_revision': '12241cbc30f20730b656db7fd5a3fa36cd420843',

  'googletest_revision': '4a00a24fff3cf82254de382437bf840cab1d3993',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '6dcd83d60f7944926bfd308cc13979fc53dd69ca',

  'spirv_headers_revision': '09913f088a1197aba4aefd300a876b2ebbaa3391',
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

