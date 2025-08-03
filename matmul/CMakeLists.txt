cmake_minimum_required(VERSION 3.17)
project(matmul_cutlass LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUTLASS directory - adjust this to your CUTLASS installation path
set(CUTLASS_DIR "/path/to/cutlass" CACHE PATH "Path to CUTLASS")
include_directories(${CUTLASS_DIR}/include)
include_directories(${CUTLASS_DIR}/tools/util/include)

# Set CUDA architecture flags based on your GPU
# Adjust these flags based on your GPU architecture
set(CUDA_ARCH_FLAGS "-gencode=arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS} -Xcompiler=-Wall")

# Add debug flags for development
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

# Add CUTLASS examples directory for linking
link_directories(${CUTLASS_DIR}/build/tools/library)

# Create the executable
add_executable(matmul_cutlass matmul_cutlass.cu)

# Link against CUDA libraries
target_link_libraries(matmul_cutlass ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

# Optional: Add an install target
install(TARGETS matmul_cutlass DESTINATION bin) 