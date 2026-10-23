cmake_minimum_required(VERSION 3.16)
project(dohc_ir_codec LANGUAGES CXX)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MPP REQUIRED IMPORTED_TARGET rockchip_mpp)
pkg_check_modules(RGA REQUIRED IMPORTED_TARGET librga)
# Some board SDKs advertise /usr/include but install MPP headers in rockchip/.
find_path(MPP_HEADER_DIR rk_mpi.h HINTS ${MPP_INCLUDE_DIRS} PATH_SUFFIXES rockchip)
if(NOT MPP_HEADER_DIR)
    message(FATAL_ERROR "MPP headers are missing from this SDK")
endif()
add_library(dohc_ir_codec SHARED codec.cpp)
target_include_directories(dohc_ir_codec PRIVATE ${MPP_HEADER_DIR})
target_compile_features(dohc_ir_codec PRIVATE cxx_std_17)
target_compile_options(dohc_ir_codec PRIVATE -Wall -Wextra -Werror)
target_link_libraries(dohc_ir_codec PRIVATE PkgConfig::MPP PkgConfig::RGA)
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE IR_CODEC_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)
target_compile_definitions(dohc_ir_codec PRIVATE
    IR_CODEC_BUILD_INFO="commit=${IR_CODEC_COMMIT},mpp=${MPP_VERSION},rga=${RGA_VERSION}")
install(TARGETS dohc_ir_codec LIBRARY DESTINATION native/infrared_codec)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/build-info.txt"
    "commit=${IR_CODEC_COMMIT},mpp=${MPP_VERSION},rga=${RGA_VERSION},compiler=${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION},processor=${CMAKE_SYSTEM_PROCESSOR}\n")

include(CTest)
if(BUILD_TESTING)
    add_executable(ir_codec_selftest selftest.cpp)
    target_include_directories(ir_codec_selftest PRIVATE ${MPP_HEADER_DIR})
    target_compile_features(ir_codec_selftest PRIVATE cxx_std_17)
    target_compile_options(ir_codec_selftest PRIVATE -UNDEBUG)
    target_link_libraries(ir_codec_selftest PRIVATE PkgConfig::MPP PkgConfig::RGA)
    add_test(NAME ir_codec_input_boundary COMMAND ir_codec_selftest)
endif()
