use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': 'c52afac4f87ef76e6293b84874e5126a62be1f15',

  'effcee_revision': '874b47102c57a8979c0f154cf8e0eab53c0a0502',

  'googletest_revision': '0bdccf4aa2f5c67af967193caf31d42d5c49bde2',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': 'c84a140c93352cdabbfb547c531be34515b12228',

  'spirv_headers_revision': '54a521dd130ae1b2f38fef79b09515702d135bdd',
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

