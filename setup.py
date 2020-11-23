from distutils.core import setup, Extension
import numpy as np

_inc_dirs = [
    '.',
    'src/',
    'src/ext/hnswlib/',
    'src/ext/tabix/',
    np.get_include()
]

_compile_args = [
    '--std=c++14',
    '-fpermissive',
    '-O3',
    '-DNDEBUG',
    '-DCPYTHON',
    '-DNPY_NO_DEPRECATED_API',
    '-Wno-sign-compare',
    '-Wno-maybe-uninitialized',
    '-Wno-unused-variable',
    '-fopenmp',
    '-msse2'
]

_link_args = [
    '-fopenmp',
]

mmutil_module = Extension(
    'mmutil',
    include_dirs=_inc_dirs,
    sources=['src/mmutil_python.cc',
             'src/utils/gzstream.cc',
             'src/utils/bgzstream.cc',
             'src/ext/tabix/bgzf.c',
             'src/ext/tabix/kstring.c'],
    language='c++',
    extra_compile_args=_compile_args,
    extra_link_args=_link_args
)

with open("README.md", "r") as fh:
    _description = fh.read()

setup(
    name="mmutil",
    version="0.2.1",
    author="Yongjin Park",
    author_email="ypp@stat.ubc.ca",
    description='matrix market utility',
    long_description=_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.5',
    ext_modules=[mmutil_module],
)
