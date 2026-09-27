cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(LIBRARY_NAME cagedeformations)
set(ALIAS_NAME CageDeformations)

project(lib${LIBRARY_NAME} VERSION 1.0.0 LANGUAGES CXX)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

set(${LIBRARY_NAME}_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}/globals.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}/GreenCoordinates.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}/LoadMesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}/LoadConstraints.h)

set(${LIBRARY_NAME}_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/globals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GreenCoordinates.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/LoadMesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/LoadConstraints.cpp)

add_library(${LIBRARY_NAME} ${${LIBRARY_NAME}_SOURCES} ${${LIBRARY_NAME}_HEADERS})

source_group(
    TREE "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${${LIBRARY_NAME}_HEADERS})

add_library(${ALIAS_NAME}::${LIBRARY_NAME} ALIAS ${LIBRARY_NAME})

target_include_directories(${LIBRARY_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${IGL_INCLUDE_DIR}
)

target_compile_features(${LIBRARY_NAME} PRIVATE cxx_auto_type)

if (MSVC)
    target_compile_options(${LIBRARY_NAME} PRIVATE /W3)
else()
    target_compile_options(${LIBRARY_NAME} PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall -Wextra -Wpedantic>)
endif()

set_target_properties(${LIBRARY_NAME} PROPERTIES DEBUG_POSTFIX "d")

target_link_libraries(${LIBRARY_NAME}
    PUBLIC
        Eigen3::Eigen
)