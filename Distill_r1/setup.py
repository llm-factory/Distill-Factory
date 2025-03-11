from setuptools import setup, find_packages

setup(
    name="distillr1",

    entry_points={
        'console_scripts': [
            'distillr1-cli=distillr1.cli:main',
        ],
    },
)