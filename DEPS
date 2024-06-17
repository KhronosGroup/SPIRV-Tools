use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '1a31b81c0a467c1c8e229b9fc172a4eb0db5bd85',

  'effcee_revision': '19b4aa87af25cb4ee779a071409732f34bfc305c',

  'googletest_revision': '1d17ea141d2c11b8917d2c7d029f1c4e2b9769b2',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '4a8cee3dd3c3d81b6fe8b867811e193d5819df07',

  'spirv_headers_revision': 'eb49bb7b1136298b77945c52b4bbbc433f7885de',
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

