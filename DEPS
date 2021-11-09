use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'effcee_revision': 'ddf5e2bb92957dc8a12c5392f8495333d6844133',
  'googletest_revision': '955c7f837efad184ec63e771c42542d37545eaef',
  're2_revision': '4244cd1cb492fa1d10986ec67f862964c073f844',
  'spirv_headers_revision': '19e8350415ed9516c8afffa19ae2c58559495a67',
}

deps = {
  'external/effcee':
      Var('github') + '/google/effcee.git@' + Var('effcee_revision'),

  'external/googletest':
      Var('github') + '/google/googletest.git@' + Var('googletest_revision'),

  'external/re2':
      Var('github') + '/google/re2.git@' + Var('re2_revision'),

  'external/spirv-headers':
      Var('github') +  '/KhronosGroup/SPIRV-Headers.git@' +
          Var('spirv_headers_revision'),
}

