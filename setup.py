from setuptools import setup, find_packages

exec(open('GASP/__version__.py').read())

setup(
    name='GASP',
    version=__version__,
    packages=find_packages(),
    description='Generalized Algorithm for Agglomerative Signed Graph Partitioning',
    author='Alberto Bailoni',
    url='https://github.com/abailoni/GASP',
    author_email='alberto.bailoni@iwr.uni-heidelberg.de',
)
