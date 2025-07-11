cmake_minimum_required(VERSION 3.15)
project(greedy_joining)

include_directories(include)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# unit tests
# add_executable(test-greedy-joining unit-tests/greedy-joining.cxx)
# add_test(test-greedy-joining test-greedy-joining)


set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(greedy_joining python-interface.cxx)
install(TARGETS greedy_joining DESTINATION .)