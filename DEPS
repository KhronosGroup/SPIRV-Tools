use_relative_paths = True

vars = {
  'chromium_git': 'https://chromium.googlesource.com',
  'github': 'https://github.com',

  'buildtools_revision': 'ab7b6a7b350dd15804c87c20ce78982811fdd76f',
  'clang_revision': 'abe5e4f9dc0f1df848c7a0efa05256253e77a7b7',
  'effcee_revision': '04b624799f5a9dbaf3fa1dbed2ba9dce2fc8dcf2',
  'googletest_revision': '98a0d007d7092b72eea0e501bb9ad17908a1a036',
  're2_revision': '6cf8ccd82dbaab2668e9b13596c68183c9ecd13f',
  'spirv_headers_revision': 'ff684ffc6a35d2a58f0f63108877d0064ea33feb',
}

deps = {
  'buildtools':
      Var('chromium_git') + '/chromium/buildtools.git@' +
          Var('buildtools_revision'),

  'external/spirv-headers':
      Var('github') +  '/KhronosGroup/SPIRV-Headers.git@' +
          Var('spirv_headers_revision'),

  'external/googletest':
      Var('github') + '/google/googletest.git@' + Var('googletest_revision'),

  'external/effcee':
      Var('github') + '/google/effcee.git@' + Var('effcee_revision'),

  'external/re2':
      Var('github') + '/google/re2.git@' + Var('re2_revision'),

  'tools/clang':
      Var('chromium_git') + '/chromium/src/tools/clang@' + Var('clang_revision')
}

recursedeps = [
  # buildtools provides clang_format, libc++, and libc++api
  'buildtools',
]

hooks = [
  # Pull clang-format binaries using checked-in hashes.
  {
    'name': 'clang_format_win',
    'pattern': '.',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=win32',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', 'SPIRV-Tools/buildtools/win/clang-format.exe.sha1',
    ],
  },
  {
    'name': 'clang_format_mac',
    'pattern': '.',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=darwin',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', 'SPIRV-Tools/buildtools/mac/clang-format.sha1',
    ],
  },
  {
    'name': 'clang_format_linux',
    'pattern': '.',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=linux*',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', 'SPIRV-Tools/buildtools/linux64/clang-format.sha1',
    ],
  },
  {
    # Pull clang
    'name': 'clang',
    'pattern': '.',
    'action': ['python',
               'SPIRV-Tools/tools/clang/scripts/update.py'
    ],
  },
]
