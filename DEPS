use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '0df56740031c3882c3b3f93bef15698a30f3a1f6',

  'effcee_revision': '2c97e5689ed8d7ab6ae5820f884f03a601ae124b',

  'googletest_revision': '6dae7eb4a5c3a169f3e298392bff4680224aa94a',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '6dcd83d60f7944926bfd308cc13979fc53dd69ca',

  'spirv_headers_revision': '2a9b6f951c7d6b04b6c21fe1bf3f475b68b84801',
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

