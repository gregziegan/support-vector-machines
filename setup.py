from setuptools import setup

setup(
    name='eecs440-ssvm',
    version='1.0',
    packages=['src', 'src.tests'],
    url='https://bitbucket.org/gregory_ziegan/eecs440-ssvm',
    license='',
    author='Greg Ziegan, MJ Harkins',
    author_email='greg.ziegan@gmail.com, mph47@case.edu',
    description='Smooth Support Vector Learning Algorithm',
    requires=['numpy', 'matplotlib', 'scipy', 'cvxopt', 'pytest']
)
