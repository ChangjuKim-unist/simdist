cmake_minimum_required(VERSION 3.8)
project(go1_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(unitree_go REQUIRED)

# unitree_legged_sdk v3.8.0 (Go1). Prebuilt static library shipped with the SDK;
# the Dockerfile clones it to /opt/unitree_legged_sdk.
if(DEFINED ENV{UNITREE_LEGGED_SDK_DIR})
  set(UNITREE_LEGGED_SDK_DIR $ENV{UNITREE_LEGGED_SDK_DIR})
else()
  set(UNITREE_LEGGED_SDK_DIR /opt/unitree_legged_sdk)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
  set(UNITREE_SDK_ARCH arm64)
else()
  set(UNITREE_SDK_ARCH amd64)
endif()
set(UNITREE_SDK_LIB ${UNITREE_LEGGED_SDK_DIR}/lib/cpp/${UNITREE_SDK_ARCH}/libunitree_legged_sdk.a)
if(NOT EXISTS ${UNITREE_SDK_LIB})
  message(FATAL_ERROR
    "unitree_legged_sdk not found at ${UNITREE_LEGGED_SDK_DIR}. "
    "Clone https://github.com/unitreerobotics/unitree_legged_sdk (tag v3.8.0) "
    "there or set UNITREE_LEGGED_SDK_DIR.")
endif()

add_executable(go1_bridge src/go1_bridge.cpp)
target_include_directories(go1_bridge PRIVATE ${UNITREE_LEGGED_SDK_DIR}/include)
target_link_libraries(go1_bridge ${UNITREE_SDK_LIB} pthread)
ament_target_dependencies(go1_bridge rclcpp rosgraph_msgs unitree_go)

install(TARGETS go1_bridge DESTINATION lib/${PROJECT_NAME})

ament_package()
