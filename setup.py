from setuptools import setup
from setuptools import find_packages

setup(name='alignment',
      version='0.0.1',
      description='Quantifying data alignment on Graph Convolutional Networks',
      author='Yifan Qian',
      author_email='haczqyf@gmail.com',
      url='https://haczqyf.github.io/',
      download_url='...',
      license='MIT',
      install_requires=['networkx','pandas','scipy','sklearn'],
      package_data={'alignment': ['README.md']},
      packages=find_packages())