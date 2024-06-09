from setuptools import setup, find_packages

setup(
    name="joule",
    version="0.1",
    package_dir={'':'src'},
    packages=find_packages(where='src'),
    # url="https://github.com/ebekkers/ponita.git",
    license="MIT",
    author="curtischong",
    author_email="curtischong5@gmail.com",
    description="learning things",
    python_requires=">=3.10.5",
)
