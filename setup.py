from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='model_building',
   version='1.0.0',
   description='The package contains useful APIs for performing exploratory data analysis, feature engineering & model building.',
   long_description=long_description,
   author='Gautam Varadarajan',
   author_email='gautam.10791@gmail.com',
   packages=['model_building'],  # same as name
   install_requires=['pandas>=2.2.2',
                     'numpy>=1.24.4',
                     'matplotlib>=3.8.3',
                     'seaborn>=0.13.2',
                     'scikit-learn>=1.4.2']  # external packages as dependencies
)
