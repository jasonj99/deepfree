#! /usr/bin/env python 
# -*- coding: utf-8 -*- 

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
 
with open('LICENSE') as fp:
    license = fp.read()
  
setup(name = "deepfree",
      version = "0.1.11",
      author = "Zhuofu Pan",
      author_email = "475366898@qq.com",
      description = "keras-style deep network package for classification and prediction",
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords = ("deep network", "deep learning", "DBN", "SAE"),
      url = "http://github.com/fuzimaoxinan/deepfree",
      maintainer='Zhuofu Pan',
      maintainer_email="475366898@qq.com",
      license=license,
      packages = find_packages(),
      install_requires = ['tensorflow>=1.10.0' ],
      platforms=['any'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      )