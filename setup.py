from setuptools import setup, find_packages

with open("./README.md") as frd:
    long_description = frd.read()

setup(
    name='nlkit',
    version='0.0.1a',
    description='easy to use nlp tools',
    long_description=long_description,
    license='MIT',
    url='https://shihanmax.github.io',
    author='shihanmax',
    author_email='shihanmax@foxmail.com',
    install_requires=[
        "numpy>=1.17.0",
        "scipy>=1.5.0",
        "matplotlib>=3.0.0",
        "torch>=1.6.0",
        "gensim>=3.7.0",
    ],
    packages=find_packages()
)
