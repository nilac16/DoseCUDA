from skbuild import setup

setup(name='DoseCUDA',
      version='1.0',
      description='GPU-based radiation dose calculation using CUDA.',
      author='Tom Hrinivich, Calin Reamy',
      author_email='DoseCUDA@gmail.com',
      packages=['DoseCUDA'],
      install_requires=[
          'numpy', 
          'pydicom', 
          'SimpleITK'
          ],
      include_package_data=True,
      package_data={'DoseCUDA': ['lookuptables/*.csv']},
      cmake_minimum_required_version='3.0',
      cmake_source_dir='dose_kernels',
      cmake_install_dir='DoseCUDA',
      cmake_args=[
          "-DCMAKE_BUILD_TYPE=Release"
          ]
      )
