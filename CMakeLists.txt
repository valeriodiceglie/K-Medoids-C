cmake_minimum_required(VERSION 3.20)
project(kmedoid-c)
add_compile_options( -ffast-math )

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

add_compile_definitions(PB_C99_STATIC_ASSERT)


#message("-- Fetching boost")

# -- Directories

# Kmedoid
set(KMEDOID_PROJECT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(KMEDOID_TEST_DIR "${KMEDOID_PROJECT_DIR}/test")
set(KMEDOID_SRC_DIR "${KMEDOID_PROJECT_DIR}/src")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/res
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# -- Project settings
file(GLOB KMEDOID_SOURCES CONFIGURE_DEPENDS
        ${KMEDOID_SRC_DIR}/*.c
        ${K_MEANS_DIR}/kmedoid.c)

file(GLOB KMEDOID_TEST_SOURCES CONFIGURE_DEPENDS
        ${KMEDOID_SRC_DIR}/*.c
        ${K_MEDOID_DIR}/kmedoid.c
        ${KMEDOID_TEST_DIR}/*.c
        ${KMEDOID_TEST_DIR}/tests/*.c)

# -- Targets
add_library(kmedoid ${KMEDOID_SOURCES})
target_link_libraries(kmedoid ${KMEDOID_LIBS})
target_include_directories(kmedoid PUBLIC ${KMEDOID_TEST_HEADERS})

add_executable(kmedoid_test ${KMEDOID_TEST_SOURCES})
target_link_libraries(kmedoid_test PRIVATE m ${KMEDOID_LIBS})
target_include_directories(kmedoid_test PUBLIC ${KMEDOID_TEST_HEADERS})

