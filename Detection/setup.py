# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:25:38 2018

@author: wnt21
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('HOGedit.pyx'))