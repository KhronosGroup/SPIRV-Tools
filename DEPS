use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '40cab244558f225cadcb943f84c5209f7ce876b1',

  'effcee_revision': '2c97e5689ed8d7ab6ae5820f884f03a601ae124b',

  'googletest_revision': '71815bbf7de6e10c11821d654a2fae2cf42de0f7',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '6dcd83d60f7944926bfd308cc13979fc53dd69ca',

  'spirv_headers_revision': '50bc4debdc3eec5045edbeb8ce164090e29b91f3',
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

