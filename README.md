# Real-time Face Recognition by using Deep Learning

## Requirments
- OpenCV 4.1.0
- CMake 3.15.5
- tensorflow
- dlib 19.19

## Installation Instructions
1. Install tensorflow
```
sudo apt-get install build-essential
git clone https://github.com/raspberrypi/tools.git tensorflow_src
cd tensorflow && ./tensorflow/lite/tools/make/download_dependencies.sh
./tensorflow/lite/tools/make/build_rpi_lib.sh

cd tensorflow/lite/tools/make/downloads/flatbuffers
mkdir build
cd build
cmake ..
make -j4
sudo make install
sudo ldconfig
```
3. Go to face_recognition folder
4. Use CMake to build program
```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
5. Run program
