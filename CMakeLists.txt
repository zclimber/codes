cmake_minimum_required(VERSION 3.17)
project(polar_coding)

set(CMAKE_CXX_STANDARD 17)

add_executable(polar_coding main.cpp base.h msf.h viterbi.h reedmuller.h testing.h polar_code.h )