import setuptools

# Package meta-data.
NAME = 'opti_models'
VERSION = '0.0.1'
AUTHOR = 'IlyaDobrynin, IvanPanshin'
EMAIL = 'iliadobrynin@yandex.ru, ivan.panshin@protonmail.com'
URL = "https://github.com/IlyaDobrynin/opti_models"
LICENSE = "MIT License"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license=LICENSE,
    scripts=['./tools/cvt-onnx', './tools/cvt-trt', './tools/bench-torch', './tools/bench-trt'],
    description="Optimized models and benchmarks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Ubuntu",
    ],
)
