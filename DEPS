use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '4eb1e49cf32ac70dd1d393812ecbcad1d90af68e',

  'effcee_revision': 'abcaf70f288ae9f7816b72b2a36fe4add4719a48',

  'googletest_revision': '90a41521142c978131f38c6da07b4eb96a9f1ff6',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': 'c84a140c93352cdabbfb547c531be34515b12228',

  'spirv_headers_revision': 'bab63ff679c41eb75fc67dac76e1dc44426101e1',
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

