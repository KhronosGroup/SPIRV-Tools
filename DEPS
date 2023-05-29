use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '79ca5d7aad63973c83a4962a66ab07cd623131ea',

  'effcee_revision': '77a53ba53bac2a5fb53b70af3cb94f177183ac29',

  'googletest_revision': '45804691223635953f311cf31a10c632553bbfc3',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '03da4fc0857c285e3a26782f6bc8931c4c950df4',
  'spirv_headers_revision': '69155b22b3b1f2d0cfed48f59167d9792de1fd79',
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

