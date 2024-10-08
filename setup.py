from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
   name='data_processing',
   version='1.0.0',
   description='The package contains useful functions for performing exploratory data analysis and feature engineering.',
   license="MIT",
   long_description=long_description,
   author='Gautam Varadarajan',
   author_email='gautam.10791@gmail.com',
   packages=['data_processing'],  # same as name
   install_requires=['pandas==2.2.2',
                     'numpy==1.24.4',
                     'matplotlib==3.8.3',
                     'seaborn==0.13.2',
                     'scikit-learn==1.4.2']  # external packages as dependencies
)
