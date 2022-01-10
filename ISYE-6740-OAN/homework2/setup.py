from setuptools import setup, find_packages
from os import path
#from pip._internal.req import parse_requirements

#reqs = parse_requirements('./requirements.txt', session=False)
#requirements = [str(ir.req) for ir in reqs]

setup(
   name='omsa_isye6740_hw2',
   version='0.0.1',
   description='Homework 2 egg',
   #long_description=readme,
   author='Mark Pearl',
   author_email='markpearl7@gmail.com',
   url='',
   #license=license,
   #install_requires=requirements,
   packages=find_packages(exclude=('jupyter', 'docs'))
)