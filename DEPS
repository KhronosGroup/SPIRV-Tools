use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '8f739d18b9d6cbf96d8143779888de0e984ba96b',

  'effcee_revision': '2c97e5689ed8d7ab6ae5820f884f03a601ae124b',

  'googletest_revision': 'd144031940543e15423a25ae5a8a74141044862f',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '6dcd83d60f7944926bfd308cc13979fc53dd69ca',

  'spirv_headers_revision': '45b314049d6262c850cc873c8f9a30f41a1e0c13',
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

