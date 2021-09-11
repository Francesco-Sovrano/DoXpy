from setuptools import setup

with open("requirements.txt", 'r') as f:
	requirements = f.readlines()

setup(
	name='knowpy',
	version='1.0',
	description='A plugin for extracting knowledge out of natural language documents.',
	url='https://www.unibo.it/sitoweb/francesco.sovrano2/en',
	author='Francesco Sovrano',
	author_email='cesco.sovrano@gmail.com',
	license='MIT',
	packages=['knowpy'],
	# zip_safe=False,
	install_requires=requirements, #external packages as dependencies
	python_requires='>=3.6',
)
