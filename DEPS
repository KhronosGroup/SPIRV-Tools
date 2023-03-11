use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'effcee_revision': '66edefd2bb641de8a2f46b476de21f227fc03a28',

  'googletest_revision': '038e392ebd8081c756e180475cc361f711fb438d',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v3.13.0.1',

  're2_revision': 'b059ae85c83ca6b1f29dba20e92e4acb85cb5b29',
  'spirv_headers_revision': '1feaf4414eb2b353764d01d88f8aa4bcc67b60db',
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

