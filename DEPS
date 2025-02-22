use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '8b68380077aacb8cda00964d865751c338e86c96',

  'effcee_revision': 'eb16bfcef98148d866e2195b86a735bb995486c7',

  'googletest_revision': '3fbe4db9a39291ae8d7a9c5f1d75896bb4c5a18f',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '6dcd83d60f7944926bfd308cc13979fc53dd69ca',

  'spirv_headers_revision': '54a521dd130ae1b2f38fef79b09515702d135bdd',
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

