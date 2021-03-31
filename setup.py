from setuptools import setup, find_packages

with open("./README.md") as frd:
    long_description = frd.read()

setup(
    name='nlkit',
    version='0.0.1a',
    description='easy to use nlp tools',
    long_description=long_description,
    license='Apache License 2.0',
    url='https://shihanmax.github.io',
    author='shihanmax',
    author_email='shihanmax@foxmail.com',
    install_requires=[],
    packages=find_packages()
)
