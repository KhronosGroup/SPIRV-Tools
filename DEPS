use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '074a32af66648c74dd0104e251e44ace5b59f7fa',

  'effcee_revision': 'd74d33d93043952a99ae7cd7458baf6bc8df1da0',

  'googletest_revision': '34ad51b3dc4f922d8ab622491dd44fc2c39afee9',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '6dcd83d60f7944926bfd308cc13979fc53dd69ca',

  'spirv_headers_revision': '41a8eb27f1a7554dadfcdd45819954eaa94935e6',
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

