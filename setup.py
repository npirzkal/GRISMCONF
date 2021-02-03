import os
from setuptools import setup, Extension, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name='grismconf',
	version='1.32',
	description='GRISM Configuration python code, described in ISR 2017-01: A More Generalized Coordinate Transformation Approach for Grisms, Pirzkal & Ryan 2017 http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2017-01.pdf',
	url='https://github.com/npirzkal/GRISMCONF',
	author='Nor Pirzkal',
	author_email='npirzkal@mac.com',
    package_dir = {
        'grismconf': 'grismconf',
        },
    packages=["grismconf"],
)
