cmake_minimum_required(VERSION 3.15.5)

# name the project
project(FaceRecognition)

# add dlib directory
add_subdirectory(../dlib dlib_build)

# add opencv library
find_package(OpenCV REQUIRED)

# target to compile
add_executable(face_recognition face_recognition.cpp)
add_executable(encode_faces encode_faces.cpp)

# add pthreads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(encode_faces Threads::Threads)

# set tflite path
set (TENSORFLOW_PATH "/home/pi/workspace/tensorflow_src")
set (TFLITE_PATH "${TENSORFLOW_PATH}/tensorflow/lite")

# link OpenCV
target_link_libraries(face_recognition ${OpenCV_LIBS})
target_link_libraries(encode_faces ${OpenCV_LIBS} dlib::dlib)

# include tflite directories
target_include_directories(encode_faces PUBLIC
	${TENSORFLOW_PATH}
	${TFLITE_PATH}
	${TFLITE_PATH}/tools/make/downloads
	${TFLITE_PATH}/tools/make/downloads/flatbuffers/include
	${TFLITE_PATH}/tools/make/gen/rpi_arm7l/lib
	)

# link tflite static library
target_link_libraries(encode_faces
	${TFLITE_PATH}/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a
	)
