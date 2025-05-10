use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '9a89ea8714c06aee702058c6acc094bff02dfd42',

  'effcee_revision': '8ce15c424e61a94ee27b5be0ec0ed036b158e6e3',

  'googletest_revision': '90a41521142c978131f38c6da07b4eb96a9f1ff6',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': 'c84a140c93352cdabbfb547c531be34515b12228',

  'spirv_headers_revision': '6d0784e9f1ab92c17eeea94821b2465c14a52be9',
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

