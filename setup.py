from setuptools import setup, find_packages

setup(
    name="terracognita",
    version="1.0.0",
    description="AI for Lost Civilization Discovery using Satellite Imagery and Multi-modal Data Fusion",
    author="mwasifanwar",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "rasterio>=1.2.0",
        "pyproj>=3.0.0",
        "matplotlib>=3.5.0",
        "folium>=0.12.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0",
    ],
    python_requires=">=3.8",
)