from setuptools import setup

with open("requirements.txt", 'r') as f:
	requirements = f.readlines()

setup(
	name='doxpy',
	version='3.0',
	description='A package for estimating the degree of explainability of information.',
	url='https://www.unibo.it/sitoweb/francesco.sovrano2/en',
	author='Francesco Sovrano',
	author_email='cesco.sovrano@gmail.com',
	license='MIT',
	packages=['doxpy'],
	# zip_safe=False,
	install_requires=requirements, #external packages as dependencies
	python_requires='>=3.6',
)
