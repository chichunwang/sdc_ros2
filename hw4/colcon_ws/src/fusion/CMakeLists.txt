cmake_minimum_required(VERSION 3.8)
project(fusion)

find_package(ament_cmake REQUIRED)

# Install Python scripts
install(PROGRAMS
  scripts/fusion.py
  DESTINATION lib/${PROJECT_NAME}
)

# Keep the student's filter module importable beside fusion.py.
install(FILES
  scripts/EKF.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install resource directories
install(DIRECTORY
  launch
  rviz_config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
