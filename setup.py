#!/usr/bin/env python3

import shutil
from setuptools import setup

setup(name='Pine',
      description='Python Neural Networks',
      author='Mike Dusenberry',
      url='https://github.com/dusenberrymw/Pine',
      packages=['pine'],
     )

shutil.rmtree('dist')
shutil.rmtree('build')
shutil.rmtree('Pine.egg-info')
