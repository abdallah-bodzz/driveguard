from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="driveguard",
    version="5.3.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real‑time driver fatigue monitoring using computer vision and CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DriveGuard",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "tensorflow>=2.13.0",
        "customtkinter>=5.2.0",
        "pygame>=2.5.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Automotive",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "driveguard = driveguard.main:main",
        ],
    },
)
