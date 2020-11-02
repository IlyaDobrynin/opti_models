import setuptools

# Package meta-data.
NAME = 'opti_models'
VERSION = '0.0.1'
AUTHOR = 'IlyaDobrynin, IvanPanshin'
EMAIL = 'iliadobrynin@yandex.ru, ivan.panshin@protonmail.com'

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description="Optimized models and benchmarks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Ubuntu",
    ],
    install_requires=[
        'tb-nightly==2.1.0a20190923',
        'future>=0.17.1',
        'opencv-python>=4.1.0.25',
        'torchvision==0.5.0',
        'torchsummary==1.5.1',
        'torch==1.4.0',
        'networkx>=2.3',
        'numpy>=1.15.4',
        'pandas>=0.23.4',
        'matplotlib>=3.0.2',
        'requests>=2.21.0',
        'albumentations==0.3.0',
        'scikit_image>=0.14.2',
        'setuptools==41.0.0',
        'tqdm>=4.28.1',
        'scikit_learn>=0.21.3',
        'pyyaml==3.13',
        'onnx>=1.4.1',
        'test-generator==0.1.2'
    ]
)
