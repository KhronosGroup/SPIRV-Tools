use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '179d2e39e79e0c313bb2cafd5784da1ce3a52f48',

  'effcee_revision': '514b52ec61609744d7e587d93a7ef9b60407ab45',

  'googletest_revision': '9df216cc9d6fd2f27e00e5ba1e88a502d7f278c7',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v21.12',

  're2_revision': '6569a9a3df256f4c0c3813cb8ee2f8eef6e2c1fb',

  'spirv_headers_revision': '01e0577914a75a2569c846778c2f93aa8e6feddd',

  'mimalloc_revision': '09a27098aa6e9286518bd9c74e6ffa7199c3f04e',
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

  'external/mimalloc':
      Var('github') + '/microsoft/mimalloc.git@' + Var('mimalloc_revision'),
}

