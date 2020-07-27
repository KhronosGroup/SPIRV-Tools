use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'effcee_revision': '2ec8f8738118cc483b67c04a759fee53496c5659',
  'googletest_revision': 'a781fe29bcf73003559a3583167fe3d647518464',
  're2_revision': 'ca11026a032ce2a3de4b3c389ee53d2bdc8794d6',
  'spirv_headers_revision': '979924c8bc839e4cb1b69d03d48398551f369ce7',
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

