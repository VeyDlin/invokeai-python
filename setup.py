# Path: setup.py
from setuptools import setup, find_packages

setup(
    name="invokeai-python",
    version="0.2.1",
    description="InvokeAI API package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Kyle Nekto",
    author_email="nekto@veydlin.com",
    url="https://github.com/veydlin/invokeai-python",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "aiohttp==3.11.11",
        "pydantic==2.10.5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
