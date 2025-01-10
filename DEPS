use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '4e0956109640efa4b2ee68a2d6640fb13afa420e',

  'effcee_revision': '08da24ec245a274fea3a128ba50068f163390565',

  'googletest_revision': '504ea69cf7e9947be54f808a09b7b08988e84b5f',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '6dcd83d60f7944926bfd308cc13979fc53dd69ca',

  'spirv_headers_revision': '3f17b2af6784bfa2c5aa5dbb8e0e74a607dd8b3b',
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

