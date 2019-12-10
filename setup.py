from distutils.core import setup, Extension

mmutil_module = Extension(
    'mmutil',
    include_dirs=['.', 'src/'],
    sources=['src/mmutil_python.cc',
             'src/utils/gzstream.cc'],
    language='c++',
    extra_compile_args=['--std=c++14', '-O3', '-DNDEBUG'],
)

setup(
    name='mmutil',
    versio='0.1.0',
    description='matrix market utility',
    ext_modules=[mmutil_module],
)
