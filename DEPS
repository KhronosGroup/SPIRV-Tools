use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': 'b4e899a95527f62cbaf6169fd7415ac944ef08af',

  'effcee_revision': 'e63a164aa0a40a04ddca2c18976819668b5a47a8',

  'googletest_revision': 'e4ece4881d1fefc1e67d21c7493835815cd13085',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '6dcd83d60f7944926bfd308cc13979fc53dd69ca',

  'spirv_headers_revision': 'a380cd25433092dbce9a455a3feb1242138febee',
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

