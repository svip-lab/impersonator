from setuptools import setup
import unittest


def test_all():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='*_test.py')
    return test_suite


setup(
    description='evaluators of Human Image Synthesize (HIS), including '
                'Motion Imitation(MI), Appearance Transfer (AT), Novel View Synthesize(NVS)',
    name='his_evaluators',
    version='0.1.0',
    author='liuwen',
    author_email='liuwen@shanghaitech.edu.cn',
    packages=['his_evaluators'],
    package_data={
        './data': ['*.json'],
    },
    license='MIT License',
    test_suite='setup.test_all',
    install_requires=[
        "scikit_image >= 0.16.2",
        "torchvision == 0.4.0",
        "torch == 1.2",
        "scipy >= 1.2.1",
        "opencv_contrib_python >= 3.4.2.17",
        "tqdm >= 4.28.1",
        "numpy >= 1.14.5",
        "setuptools >= 39.1.0",
        "Pillow >= 6.2.0",
        "typing >= 3.7.4.1"
    ],
)
