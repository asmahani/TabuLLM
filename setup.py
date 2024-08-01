from setuptools import setup, find_packages

setup(
    name='TabuLLM',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here, e.g.,
        'numpy'
        , 'openai'
        , 'pandas'
        , 'scikit-learn'
        , 'scipy'
        , 'sentence-transformers'
        , 'gensim'
        , 'vertexai'
    ],
    # Additional metadata about your package.
    author='Alireza S. Mahani',
    author_email='alireza.s.mahani@gmail.com',
    description='A Python package for feature extraction from text columns in tabular data using large language models (LLMs).',
    license='MIT',
    keywords=['text embedding', 'large language models', 'feature engineering', 'cross-validation'],
)
