use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'effcee_revision': '66edefd2bb641de8a2f46b476de21f227fc03a28',

  'googletest_revision': '88af49efa72a06d56910d69ecfd87e3b330e5778',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v3.13.0.1',

  're2_revision': '11073deb73b3d01018308863c0bcdfd0d51d3e70',
  'spirv_headers_revision': '90547c54e24e01aae41a68124e7a304d0ec11dd0',
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

