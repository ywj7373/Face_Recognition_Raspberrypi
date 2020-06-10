#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/tools/gen_op_registration.h>
#include <tensorflow/lite/kernels/register.h>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace tflite;
using namespace std;
using namespace cv;

vector<float> face_encoding(char* modelPath, Mat img);

bool match(vector<float> img1, vector<float> img2, float threshold);

int get_embedded_size(char* modelPath);


