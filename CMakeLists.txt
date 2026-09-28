cmake_minimum_required(VERSION 3.16)
project(euchre C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(euchre_engine src/euchre.c)
target_include_directories(euchre_engine PUBLIC include)
target_compile_options(euchre_engine PRIVATE -Wall -Wextra -Wpedantic)

add_executable(euchre_demo src/main.c)
target_link_libraries(euchre_demo PRIVATE euchre_engine)
target_compile_options(euchre_demo PRIVATE -Wall -Wextra -Wpedantic)
