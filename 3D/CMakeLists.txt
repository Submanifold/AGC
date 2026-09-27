cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cageDeformation3D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_definitions(EIGEN_DONT_VECTORIZE)
endif()


add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/external/Eigen3")


set(IGL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/libigl/include")

# ---------- cagedeformations ----------
add_subdirectory(cagedeformations)

add_executable(cageDeformation3D main.cpp)

target_include_directories(cageDeformation3D PRIVATE ${IGL_INCLUDE_DIR})

target_link_libraries(cageDeformation3D
    PRIVATE
    cagedeformations
    Eigen3::Eigen)

install(TARGETS cageDeformation3D RUNTIME DESTINATION bin)