use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '35d35604d6d57850699b5a8cb9da2a5ea418ff88',

  'effcee_revision': '12241cbc30f20730b656db7fd5a3fa36cd420843',

  'googletest_revision': '54501746a6cfd366b6e7bab3c902e0f8d5826521',

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

