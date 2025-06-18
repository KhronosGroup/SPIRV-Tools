use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': 'f8288c18a12bbe158dcdee5cb9030efd109e7858',

  'effcee_revision': '8ce15c424e61a94ee27b5be0ec0ed036b158e6e3',

  'googletest_revision': 'fd15f51d57f983c5f3f609bb39fd77f6dbdc391a',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': 'c84a140c93352cdabbfb547c531be34515b12228',

  'spirv_headers_revision': '4e209d3d7e555d49b9a4720a622065881f839ac8',
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

