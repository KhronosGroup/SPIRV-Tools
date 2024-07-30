use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '52fad5aa026b6202dd8dabb5723f2ee3f604048f',

  'effcee_revision': '2c97e5689ed8d7ab6ae5820f884f03a601ae124b',

  'googletest_revision': '5bcb2d78a16edd7110e72ef694d229815aa29542',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '6dcd83d60f7944926bfd308cc13979fc53dd69ca',

  'spirv_headers_revision': 'f013f08e4455bcc1f0eed8e3dd5e2009682656d9',
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

