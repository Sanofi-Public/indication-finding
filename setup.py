from setuptools import setup, find_packages

# Read the contents of your requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='IndicationID',
    version='0.1.0',
    author='Lukas Adamek',
    author_email='lukas.adamek@sanofi.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    description='A package that handles matrix factorization ',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/yourusername/yourpackagename',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
)
