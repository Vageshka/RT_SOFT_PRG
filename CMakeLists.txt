cmake_minimum_required(VERSION 3.10)
SET("OpenCV_DIR" "/opencv_build/opencv/")
project(code)
find_package( OpenCV REQUIRED )

MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

add_executable(code main.cpp)
target_link_libraries(code ${OpenCV_LIBS})
