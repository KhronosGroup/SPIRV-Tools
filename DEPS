use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'effcee_revision': '66edefd2bb641de8a2f46b476de21f227fc03a28',

  'googletest_revision': '48a1b110583dd55e5076952b2acd772d9aaf6372',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v3.13.0.1',

  're2_revision': 'f0402c0397cd4cfba17203abd5b617558316af0c',
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

