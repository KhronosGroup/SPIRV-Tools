use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '1315c900e1ddbb08a23e06eeb9a06450052ccb5e',

  'effcee_revision': 'd74d33d93043952a99ae7cd7458baf6bc8df1da0',

  'googletest_revision': '1d17ea141d2c11b8917d2c7d029f1c4e2b9769b2',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '4a8cee3dd3c3d81b6fe8b867811e193d5819df07',

  'spirv_headers_revision': '2acb319af38d43be3ea76bfabf3998e5281d8d12',
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

