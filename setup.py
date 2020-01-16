from setuptools import setup, find_packages

__version__ = '1.0.0'

setup(
    name='GASP',
    version=__version__,
    packages=find_packages(),
    description='Generalized Algorithm for Agglomerative Signed Graph Partitioning',
    author='Alberto Bailoni',
    url='https://github.com/abailoni/GASP',
    # long_description='',
    author_email='alberto.bailoni@iwr.uni-heidelberg.de',
    # install_requires=['numpy'],
)
