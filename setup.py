from setuptools import setup

__version__ = '0.1'

setup(
    name='GASP',
    version=__version__,
    packages=['GASP', ],
    description='Generalized Algorithm for Agglomerative Signed Graph Partitioning',
    author='Alberto Bailoni',
    url='https://github.com/abailoni/GASP',
    # long_description='',
    author_email='alberto.bailoni@iwr.uni-heidelberg.de',
    install_requires=['numpy', 'nifty<='], #TODO: add nifty version
)
