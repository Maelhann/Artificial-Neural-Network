cmake_minimum_required(VERSION 3.10)
project(ann C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(ann
        ann.c
        ann.h
        layer.c
        layer.h
        rdata.c
        train.c)
