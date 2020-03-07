cmake_minimum_required(VERSION 2.8)

PROJECT(HeadCTReconstruction)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

add_executable(HeadCTReconstruction MACOSX_BUNDLE HeadCTReconstruction.cxx )

target_link_libraries(HeadCTReconstruction ${VTK_LIBRARIES})