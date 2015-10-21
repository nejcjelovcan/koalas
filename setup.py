# coding=utf-8

import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
  return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = "koalas",
  version = "0.0.1",
  author = "Nejc Jelovčan",
  author_email = "nejc.jelovcan@gmail.com",
  description = "pandas toolkit",
  license = "BSD",
  keywords = "pandas dataframe toolkit",
  url = "http://example.com",
  packages=['koalas'],
  #packages=['an_example_pypi_project', 'tests'],
  long_description=read('README'),
  classifiers=[
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "License :: OSI Approved :: BSD License",
  ],
  test_suite="tests.suite"
)
