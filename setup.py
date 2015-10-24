from __future__ import absolute_import
from setuptools import setup, Extension

setup(
    name='stm',
    version='1.0.0',
    description='Structural Topic Model in Python',
    author='Antonio Coppola, Margaret E. Roberts, Brandon M. Stewart and Dustin Tingley',
    author_email="acoppola@college.harvard.edu, bms4@princeton.edu",
    packages=['stm'],
    install_requires=['numpy>=1.9', 'six'],
    setup_requires=['numpy>=1.9'],
    keywords=['Topic models', 'natural language processing', 
    'unsupervised learning', 'machine learning', 'LDA', 
    'Python', 'Numpy', 'Scipy'],
    url='https://github.com/AntonioCoppola/stm-spark-dev',
    license='MIT',
    classifiers=['Programming Language :: Python :: 2.7'],
    ext_modules=extensions,
)
