from setuptools import setup,find_packages

setup(name="torchfusion",
      version='0.3.5',
      description='A modern deep learning framework built to accelerate research and development of AI systems',
      url="https://github.com/johnolafenwa/TorchFusion",
      author='John Olafenwa and Moses Olafenwa',
      license='MIT',
      packages= find_packages(),
      install_requires=['torchvision','torchtext','numpy','matplotlib',"tqdm","tensorboardX","visdom"],
      zip_safe=False,
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
      )
