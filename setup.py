from setuptools import find_packages, setup

setup(
    name="fsdatasets",
    version="0.0.1",
    packages=find_packages(include=["fsdatasets", "fsdatasets.*"]),
    install_requires=["numpy", "scipy", "torch"],
)
