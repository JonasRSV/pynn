import setuptools

setuptools.setup(
    name="nn",
    version="0.0.1",
    author="Jonas",
    author_email="jonas@valfridsson.net",
    description="Simple NN implementation in python",
    url="",
    packages=setuptools.find_packages(exclude=["tests"]),
    install_requires=["numpy", "halo"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
