use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '8b2b78bb9bc570ea422fc891392d2ebdb35468b1',

  'effcee_revision': '874b47102c57a8979c0f154cf8e0eab53c0a0502',

  'googletest_revision': '155b337c938a2953e5675f9dc18c99f05f4c85d0',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': 'c84a140c93352cdabbfb547c531be34515b12228',

  'spirv_headers_revision': 'aa6cef192b8e693916eb713e7a9ccadf06062ceb',
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

