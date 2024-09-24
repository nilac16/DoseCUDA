cmake_minimum_required(VERSION 3.16)

project(dose_kernels C CXX CUDA)

set(MODULE_NAME ${PROJECT_NAME})

find_package(PythonExtensions REQUIRED)
find_package(NumPy REQUIRED)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_library(${MODULE_NAME} MODULE ${SOURCES})
python_extension_module(${MODULE_NAME})

target_sources(${MODULE_NAME}
    PRIVATE
        dosemodule.cpp
        BaseClasses.cpp
        CudaClasses.cu
        IMPTClasses.cu)

target_include_directories(${MODULE_NAME}
    PRIVATE
        ${NumPy_INCLUDE_DIRS})

set_target_properties(${MODULE_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE   ON
        CUDA_SEPARABLE_COMPILATION  ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_definitions(${MODULE_NAME}
    PRIVATE
        DOSECUDA_DEVICE_POINTER)

install(TARGETS ${MODULE_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX})