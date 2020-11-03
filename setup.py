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
        'opencv-python==4.1.0.25',
        'torchvision==0.8.1',
        'torchsummary==1.5.1',
        'torch==1.7.0',
        'numpy>=1.15.4',
        'pandas>=0.23.4',
        'requests>=2.21.0',
        'albumentations>=0.4.0',
        'scikit_image>=0.14.2',
        'tqdm>=4.28.1',
        'onnx>=1.7.0',
        'onnxruntime>=1.5.2',
        'pycuda==2020.1',
        'onnx-simplifier>=0.2.18'
    ]
)
