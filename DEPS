use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'effcee_revision': '66edefd2bb641de8a2f46b476de21f227fc03a28',

  'googletest_revision': '797b0ad2a3a45608ecf5c67e6e289d377a3521ca',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '11073deb73b3d01018308863c0bcdfd0d51d3e70',
  'spirv_headers_revision': '7f1d2f4158704337aff1f739c8e494afc5716e7e',
}

deps = {
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

