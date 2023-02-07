use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'effcee_revision': 'c7b4db79f340f7a9981e8a484f6d5785e24242d1',

  'googletest_revision': '3d568bdda59a0b5e50d5f08038eb092b6d88e309',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v3.13.0.1',

  're2_revision': 'b025c6a3ae05995660e3b882eb3277f4399ced1a',
  'spirv_headers_revision': 'aa331ab0ffcb3a67021caa1a0c1c9017712f2f31',
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

