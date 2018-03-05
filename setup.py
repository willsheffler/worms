#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'homog >= 0.1.10',
    'tqdm',
]

setup_requirements = [
]

test_requirements = [
    'pip',
    'wheel',
    'watchdog',
    'flake8',
    'tox',
    'coverage',
    'PyYAML',
    'pytest',
    'pytest-runner',
    'numpy',
    'homog >= 0.1.10',
    'tqdm',
    # https://storage.googleapis.com/protein-design-ipd-public/wheelhouse/pyrosetta-2017.48.post0.dev0%2B93.fordas.dev.f126926bdc8-cp35-cp35m-linux_x86_64.whl
    'pytest',
]

setup(
    name='worms',
    version='0.1.15',
    description="Protion Origami via Genetic Fusions",
    long_description=readme + '\n\n' + history,
    author="Will Sheffler",
    author_email='willsheffler@gmail.com',
    url='https://github.com/willsheffler/worms',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=True,
    keywords='worms',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='worms/tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
