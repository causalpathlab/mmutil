from distutils.core import setup, Extension
import numpy as np
import scipy as sp

_inc_dirs = [
    '.',
    'src/',
    np.get_include()
]

_compile_args = [
    '--std=c++17',
    '-O3',
    '-DNDEBUG',
    '-DNPY_NO_DEPRECATED_API',
    '-Wno-sign-compare',
    '-Wno-unused-variable'
]


mmutil_module = Extension(
    'mmutil',
    include_dirs=_inc_dirs,
    sources=['src/mmutil_python.cc',
             'src/utils/gzstream.cc'],
    language='c++',
    extra_compile_args=_compile_args,
)


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="mmutil",
    version="0.1.0",
    author="Yongjin Park",
    author_email="yongjin.peter.park@gmail.com",
    description='matrix market utility',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.5',
    ext_modules=[mmutil_module],
)
