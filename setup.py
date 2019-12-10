from distutils.core import setup, Extension
import numpy as np
import scipy as sp

_inc_dirs = [
    '.',
    'src/',
    np.get_include()
]

_compile_args = [
    '--std=c++14', '-O3', '-DNDEBUG'
]


mmutil_module = Extension(
    'mmutil',
    include_dirs=_inc_dirs,
    sources=['src/mmutil_python.cc',
             'src/utils/gzstream.cc'],
    language='c++',
    extra_compile_args=_compile_args,
)

setup(
    name='mmutil',
    versio='0.1.0',
    description='matrix market utility',
    ext_modules=[mmutil_module],
)
