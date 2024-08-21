from setuptools import setup, find_packages

setup(
    name='amazan-predictions',  # Your project name
    version='0.1.0',  # Your project version
    author='Razie',  # Your name
    description='Predicts prices and quantities using XGBoost',  # Short description
    packages=find_packages(),  # Automatically find packages
    install_requires=[
        'pandas',
        'xgboost',
        'scikit-learn',
    ],
    python_requires='>=3.8',  # Minimum Python version
)
