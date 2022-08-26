import os
from setuptools import find_namespace_packages
from setuptools import setup




setup(
    name="hydra-optuna-sweeper",
    version="0.0.1",
    author="Toshihiko Yanae",
    author_email="toshihiko.yanase@gmail.com",
    description="Hydra Optuna Sweeper plugin",
    long_description_content_type="text/markdown",
    url="https://github.com/toshihikoyanase/hydra-optuna-plugin/",
    packages=find_namespace_packages(include=["hydra_plugins.*"]),
    classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Operating System :: OS Independent",
            "Development Status :: 4 - Beta",
    ],
    install_requires=["hydra-core", "optuna"],
    include_package_data=True,
)