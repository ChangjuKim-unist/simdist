cmake_minimum_required(VERSION 3.8)
project(go1_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf DESTINATION share/${PROJECT_NAME})
# optional, see setup/fetch_go1_meshes.sh
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/meshes)
  install(DIRECTORY meshes DESTINATION share/${PROJECT_NAME})
endif()

ament_package()
