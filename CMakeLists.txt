cmake_minimum_required(VERSION 3.2)

project(hex-snake)

set(CMAKE_CXX_STANDARD 14)

find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)

add_executable(hex-snake Game.cpp data_path.cpp main.cpp)

target_include_directories(hex-snake PUBLIC ${OPENGL_INCLUDE_DIR} ${SDL2_INCLUDE_DIRS})

target_link_libraries(hex-snake ${OPENGL_LIBRARIES} ${SDL2_LIBRARIES})