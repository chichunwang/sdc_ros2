cmake_minimum_required(VERSION 3.18)
project(cuda_ndt_backend LANGUAGES C CXX CUDA)

find_package(ament_cmake REQUIRED)
find_package(PCL REQUIRED COMPONENTS common registration)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit 12.8 REQUIRED)
include(FetchContent)

# Upstream's old FindCUDA build enables unrelated VGICP/cuBLAS dependencies.
# Compile only its NDT sources with modern CMake, Eigen and CUDA 12.8 instead.
FetchContent_Declare(fast_gicp_source
  URL https://codeload.github.com/koide3/fast_gicp/tar.gz/0e7ec1441c99f7be453db2ea216d5de029387417
  URL_HASH SHA256=b18c904bbd47c8653a9df4c25aec4bf7cf19522338d63a3d155a20a46a088861
)
FetchContent_GetProperties(fast_gicp_source)
if(NOT fast_gicp_source_POPULATED)
  FetchContent_Populate(fast_gicp_source)
endif()
set(upstream "${fast_gicp_source_SOURCE_DIR}")

# CCCL 2.x moved Thrust types into an ABI namespace. Preserve upstream's
# lightweight forward declarations using Thrust's official namespace macros.
set(ndt_header "${upstream}/include/fast_gicp/cuda/ndt_cuda.cuh")
file(READ "${ndt_header}" ndt_header_text)
set(ndt_header_original "${ndt_header_text}")
string(REPLACE "namespace thrust {" "#include <thrust/detail/config.h>\nTHRUST_NAMESPACE_BEGIN" ndt_header_text "${ndt_header_text}")
string(REPLACE "}  // namespace thrust" "THRUST_NAMESPACE_END" ndt_header_text "${ndt_header_text}")
if(NOT ndt_header_original STREQUAL ndt_header_text)
  file(WRITE "${ndt_header}" "${ndt_header_text}")
endif()

add_library(cuda_ndt_backend SHARED
  src/nearest_neighbor.cu
  ${upstream}/src/fast_gicp/gicp/lsq_registration.cpp
  ${upstream}/src/fast_gicp/ndt/ndt_cuda.cpp
  ${upstream}/src/fast_gicp/cuda/ndt_cuda.cu
  ${upstream}/src/fast_gicp/cuda/ndt_compute_derivatives.cu
  ${upstream}/src/fast_gicp/cuda/gaussian_voxelmap.cu
  ${upstream}/src/fast_gicp/cuda/find_voxel_correspondences.cu
  ${upstream}/src/fast_gicp/cuda/covariance_regularization.cu
)
set_target_properties(cuda_ndt_backend PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)
target_compile_options(cuda_ndt_backend PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_include_directories(cuda_ndt_backend PUBLIC
  $<BUILD_INTERFACE:${upstream}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(cuda_ndt_backend PUBLIC ${PCL_LIBRARIES} Eigen3::Eigen CUDA::cudart)

add_executable(cuda_ndt_smoke src/cuda_ndt_smoke.cpp)
target_compile_features(cuda_ndt_smoke PRIVATE cxx_std_17)
target_link_libraries(cuda_ndt_smoke PRIVATE cuda_ndt_backend)
add_executable(nearest_neighbor_test src/nearest_neighbor_test.cpp)
target_link_libraries(nearest_neighbor_test PRIVATE cuda_ndt_backend)

install(TARGETS cuda_ndt_backend EXPORT cuda_ndt_backendTargets LIBRARY DESTINATION lib)
install(TARGETS cuda_ndt_smoke DESTINATION lib/${PROJECT_NAME})
install(TARGETS nearest_neighbor_test DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY ${upstream}/include/fast_gicp DESTINATION include)
install(FILES ${upstream}/LICENSE DESTINATION share/${PROJECT_NAME} RENAME FAST_GICP_LICENSE)
ament_export_targets(cuda_ndt_backendTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(PCL Eigen3 CUDAToolkit)
ament_package()
