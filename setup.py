import setuptools

def readme():
    with open('README.md', 'r', encoding='utf-8') as f_readme:
        return f_readme.read()

def requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f_requirements:
        return f_requirements.read()

setuptools.setup(
    name="human-detection",
    version="0.1",
    author="k-vashpanova",
    description="Simple human detection program",
    url="https://github.com/k-vashpanova/human-detection/",
    py_modules=['detect_humans'],
    install_requires=requirements(),
    long_description=readme(),
    entry_points={
        'console_scripts': [
            'detect_humans=detect_humans:main',
        ],
    },
    packages=['human-detection'],
)
