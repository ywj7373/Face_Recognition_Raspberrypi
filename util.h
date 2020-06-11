#ifndef _UTILS_FOR_PROJECTS_DEFINED
#define _UTILS_FOR_PROJECTS_DEFINED

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void get_img_list(const char* filename, vector<string> &imglist);

float* get_faces(vector<string> imglist);

void face_detect(Mat &img, Mat &face);

#endif
