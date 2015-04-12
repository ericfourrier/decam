from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='decam',
      version= "0.1",
      description='decam is a package for data cleaning exploration and modelling in pandas and scikit-learn',
      long_description=readme(),
      author=['Eric Fourrier','Leonard Berrada','Kevin Olivier'],
      author_email='ericfourrier0@gmail.com',
      license = 'MIT',
      url='https://github.com/ericfourrier/decam',
      packages=['decam'],
      test_suite = 'test',
      keywords=['cleaning','modeling', 'pandas','scikit-learn','prediction'],
      install_requires=[
          'numpy>=1.7.0',
          'pandas>=0.15.0',
          'scikit-learn>=0.14']
)