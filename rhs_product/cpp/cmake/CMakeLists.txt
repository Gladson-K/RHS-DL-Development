cmake_minimum_required(VERSION 3.10)
project(RHSGPBeamformer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
if(NOT DEFINED EIGEN3_INCLUDE_DIR)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Create library
add_library(rhs_gp_beamformer STATIC
    ../src/rhs_gp_beamformer.cpp
)

if(TARGET Eigen3::Eigen)
    target_link_libraries(rhs_gp_beamformer PUBLIC Eigen3::Eigen)
endif()

# Create executable
add_executable(rhs_gp_test ../src/main.cpp)
target_link_libraries(rhs_gp_test PRIVATE rhs_gp_beamformer)

# Installation
install(TARGETS rhs_gp_beamformer rhs_gp_test
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
