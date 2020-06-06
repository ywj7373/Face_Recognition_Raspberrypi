cmake_minimum_required(VERSION 3.15.5)

# name the project
project(FaceRecognition)

# add opencv library
find_package(OpenCV REQUIRED)

# add dlib library
add_subdirectory(./dlib dlib_build)

# target to compile
add_executable(face_recognition face_detection.cpp)

# link cpp to dlib
target_link_libraries(face_recognition dlib::dlib ${OpenCV_LIBS})
