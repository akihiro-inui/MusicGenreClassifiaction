from distutils.core  import setup, Extension
from Cython.Build import cythonize


# Cython
external_module = Extension("speedup", sources=["speedup.pyx"])


setup(
    name='MusicGenreClassifiaction',
    ext_modules=cythonize([external_module]),
    version='0.0',
    packages=['', 'utils', 'common', 'classifier', 'preprocess', 'data_process', 'feature_extraction'],
    package_dir={'': 'backend'},
    url='',
    license='',
    author='Akihiro Inui',
    author_email='mail@akihiroinui.com',
    description='Contend-based Music Genre Classification'
)