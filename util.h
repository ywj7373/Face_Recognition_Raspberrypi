#ifndef _UTILS_FOR_PROJECTS_DEFINED
#define _UTILS_FOR_PROJECTS_DEFINED

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

void get_img_list(char* filename, vector<string> &imglist);

float* get_faces(vector<string> imglist);

#endif
