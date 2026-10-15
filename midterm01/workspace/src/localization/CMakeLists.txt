cmake_minimum_required(VERSION 3.8)
project(localization)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

add_executable(localization src/localization.cpp)
ament_target_dependencies(localization
  rosbag2_cpp
  rosbag2_storage
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  pcl_conversions
)
target_link_libraries(localization
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  Boost::filesystem
  Boost::system
)

add_executable(map_modified src/map_modified.cpp)
ament_target_dependencies(map_modified
  rclcpp
)
target_link_libraries(map_modified
  ${PCL_LIBRARIES}
)

add_executable(visualization src/visualization.cpp)
ament_target_dependencies(visualization
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  pcl_conversions
)
target_link_libraries(visualization
  ${PCL_LIBRARIES}
)

install(
  TARGETS
    localization
    map_modified
    visualization
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
