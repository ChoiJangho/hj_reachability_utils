import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hj_reachability_utils",
    version="0.1.0",
    author="Jason J. Choi",
    author_email="jason.choi@berkeley.edu",
    description="Utility functions for hj_reachability toolbox.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChoiJangho/hj_reachability_utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)