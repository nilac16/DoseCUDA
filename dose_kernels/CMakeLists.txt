cmake_minimum_required(VERSION 3.16)

project(dose_kernels C CXX CUDA)

set(MODULE_NAME ${PROJECT_NAME})

add_library(${MODULE_NAME} MODULE)


find_package(PythonExtensions)

if (${PythonExtensions_FOUND})
    python_extension_module(${MODULE_NAME})
endif()


find_package(Python REQUIRED COMPONENTS NumPy Development.Module)

list(APPEND DK_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                            ${Python_INCLUDE_DIRS}
                            ${Python_NumPy_INCLUDE_DIRS})


if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()


target_sources(${MODULE_NAME}
    PRIVATE
        dosemodule.cpp
        BaseClasses.cpp
        CudaClasses.cu
        IMPTClasses.cu)

target_include_directories(${MODULE_NAME}
    PRIVATE
        ${DK_INCLUDE_DIRS})

set_target_properties(${MODULE_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE   ON
        CUDA_SEPARABLE_COMPILATION  ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(${MODULE_NAME} Python::NumPy)

target_compile_definitions(${MODULE_NAME}
    PRIVATE
        DOSECUDA_DEVICE_POINTER)

install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION DoseCUDA)
