from skbuild import setup

setup(
    name="DoseCUDA",
    version="1.0", 
    packages=['DoseCUDA'],
    include_package_data=True,
    package_data={'DoseCUDA': ['lookuptables/*.csv']},
    cmake_source_dir='dose_kernels',
    cmake_install_dir='DoseCUDA',
    cmake_args=[
          "-DCMAKE_BUILD_TYPE=Release"
          ]
      )
