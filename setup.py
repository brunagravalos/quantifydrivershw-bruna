from setuptools import find_packages, setup

setup(
    name='QuantifyDrivers', 
    version='0.1.0',         
    url='Implementation of a methodology using explainable ML to quantify drivers of temperature extremes ona daily basis',
    description="A description of what QuantifyDrivers does.",
    install_requires=[],
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)