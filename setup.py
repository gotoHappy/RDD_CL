from setuptools import find_packages, setup


setup(
    name="robust-defect-detection-cl",
    version="0.1.0",
    description="Contrastive-learning-only DINOv2 defect detection project.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "matplotlib",
        "numpy<2.0.0",
        "Pillow",
        "pyyaml",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "tqdm",
        "wandb",
        "xformers==0.0.29.post2",
    ],
)
