import setuptools


with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="tsaugur",
    version="0.0.1",
    author="Michał Oleszak",
    author_email="oleszak.michal@gmail.com",
    description="Time Series Augur: a low-code Python package for time series forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichalOleszak/tsaugur",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
