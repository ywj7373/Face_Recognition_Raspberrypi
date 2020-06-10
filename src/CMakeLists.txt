cmake_minimum_required(VERSION 3.15.5)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(encode_faces 
	encode_faces.cpp
	face_recognition.cpp
	utils.cpp
)

add_executable(main
	main.cpp
	face_recognition.cpp
	utils.cpp
)

# set tflite path
set (TENSORFLOW_PATH "/home/pi/workspace/tensorflow_src")
set (TFLITE_PATH "${TENSORFLOW_PATH}/tensorflow/lite")

set(LIBS
	Threads::Threads
	${TFLITE_PATH}/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a
	dlib::dlib
)

set(DIRS
	${TENSORFLOW_PATH}
	${TFLITE_PATH}
	${TFLITE_PATH}/tools/make/downloads
	${TFLITE_PATH}/tools/make/downloads/flatbuffers/include
	${TFLITE_PATH}/tools/make/gen/rpi_arm7l/lib
)

# add dlib header
add_subdirectory(../../dlib dlib_build)

# add opencv header
find_package(OpenCV REQUIRED)

# add pthreads header
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# add header
target_include_directories(encode_faces PUBLIC ${DIRS})
target_include_directories(main PUBLIC ${DIRS})

# add libraries
target_link_libraries(encode_faces PUBLIC ${LIBS} ${OpenCV_LIBS})
target_link_libraries(main PUBLIC ${LIBS} ${OpenCV_LIBS})
