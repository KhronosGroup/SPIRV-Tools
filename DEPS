use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'effcee_revision': '66edefd2bb641de8a2f46b476de21f227fc03a28',

  'googletest_revision': '974e18ee6f146a2418f9cea83170c640e7d622d6',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v3.13.0.1',

  're2_revision': '11073deb73b3d01018308863c0bcdfd0d51d3e70',
  'spirv_headers_revision': 'a41cf67dfa52d33c049dc1b606e2db5cf72dafeb',
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

