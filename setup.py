from setuptools import setup, find_packages

setup(
    name='wave2midi',
    version='0.1.0',
    py_modules=['wave2midi'],
    install_requires=[
        'numpy',
        'librosa',
        'mido',
        'torch',
        'torchaudio',
        'demucs @ git+https://github.com/adefossez/demucs.git',
        'dora-search',
        'einops',
        'julius>=0.2.3',
        'lameenc>=1.2',
        'openunmix',
        'pyyaml',
        'tqdm',
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-mock',
            'scipy',
        ],
    },
    entry_points={
        'console_scripts': [
            'wave2midi = wave2midi:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A tool to convert WAV files to MIDI by separating stems first.',
    license='MIT',
)
