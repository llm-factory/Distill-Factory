from setuptools import setup, find_packages

setup(
    name="llamafeeder",

    entry_points={
        'console_scripts': [
            'llamafeeder-cli=llamafeeder.cli:main',
        ],
    },
)