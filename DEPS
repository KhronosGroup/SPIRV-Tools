use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '79ca5d7aad63973c83a4962a66ab07cd623131ea',

  'effcee_revision': '6d3b974a7779506b59d70cc7ecea1e47931c7183',

  'googletest_revision': '9b12f749fa972d08703d8459e9bf3239617491ca',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '7c5e396af825562ec8321fdbf2f1cf276b26e3ae',
  'spirv_headers_revision': '6e09e44cd88a5297433411b2ee52f4cf9f50fa90',
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

