from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='GASP',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Generalized Algorithm for Signed graph Partitioning",
    license="MIT",
    author="Alberto Bailoni",
    author_email='alberto.bailoni@iwr.uni-heidelberg.de',
    url='https://github.com/abailoni/GASP',
    packages=['GASP'],
    
    install_requires=requirements,
    keywords='GASP',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
