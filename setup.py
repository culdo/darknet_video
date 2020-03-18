from setuptools import setup, find_packages
setup(
    name="DarknetWrapper",
    version="0.1",
    packages=find_packages(), install_requires=['cv2', 'numpy', 'requests']
)
