from setuptools import setup,find_packages

setup(name="torchfusion",
      version='0.2.0-rc38',
      description='A modern deep learning framework built to accelerate research and development of AI systems',
      url="https://commons.specpal.science",
      author='John Olafenwa and Moses Olafenwa',
      license='MIT',
      packages= find_packages(),
      install_requires=['torchvision','torchtext','numpy','matplotlib',"tqdm","tensorboardX","visdom"],
      zip_safe=False
      )