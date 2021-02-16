import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pointproc", # Replace with your own username
    version="0.0.1",
    author="Tomas Barta",
    author_email="tomas.barta@fgu.cas.cz",
    description="Package to work with inhomogenous point processes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tom83B/pointproc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)