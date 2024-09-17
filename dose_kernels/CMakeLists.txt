cmake_minimum_required(VERSION 3.16)

project(dose_kernels)
enable_language(CUDA)

find_package(Python3 COMPONENTS Development REQUIRED)
find_package(CUDA REQUIRED)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(SOURCES
    dosemodule.cpp
    BaseClasses.cpp
    CudaClasses.cu
    IMPTClasses.cu
)

set_source_files_properties(CudaClasses.cu IMPTClasses.cu PROPERTIES LANGUAGE CUDA)

add_library(dose_kernels SHARED ${SOURCES})

set_target_properties(dose_kernels PROPERTIES PREFIX "")

target_include_directories(dose_kernels PUBLIC ${CMAKE_SOURCE_DIR}/include ${Python3_INCLUDE_DIRS})

set_target_properties(dose_kernels PROPERTIES
        POSITION_INDEPENDENT_CODE   ON
        CUDA_SEPARABLE_COMPILATION  ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_definitions(dose_kernels
    PRIVATE
        DOSECUDA_DEVICE_POINTER)

target_link_libraries(dose_kernels ${CUDA_LIBRARIES} ${Python3_LIBRARIES})

install(TARGETS dose_kernels DESTINATION ${CMAKE_INSTALL_PREFIX})