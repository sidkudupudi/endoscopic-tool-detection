cmake_minimum_required(VERSION 3.10)
project(status_relay)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(status_relay main.cpp)
target_link_libraries(status_relay pthread)
