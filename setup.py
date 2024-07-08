from setuptools import setup, find_packages
setup(name='mml',
      version='1.0',
      description='multimodal machine learning for pain recognition module',
      author='Furkan Sevinc',
      package_dir={'': 'src'},
      packages=find_packages(where='src'))