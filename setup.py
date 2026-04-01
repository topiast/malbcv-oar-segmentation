from setuptools import setup, find_packages

setup(
    name="btcv-organ-segmentation",
    version="0.1.0",
    description="3D U-Net for automatic segmentation of abdominal organs in CT images (BTCV dataset)",
    author="Topi Astikainen",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "monai>=1.3.0",
        "SimpleITK>=2.3.0",
        "nibabel>=5.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
