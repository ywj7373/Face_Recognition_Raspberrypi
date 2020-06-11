#include <iostream>
#include <fstream>
#include "util.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string.h>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

void get_img_list(const char* filename, vector<string> &imglist){
  ifstream f;
  f.open(filename);
  string buffer;
  while(f.peek() != EOF){
    getline(f, buffer);
    imglist.push_back(buffer);
  }
}

float* get_faces(vector<string> imglist){
  float* faces = malloc(imglist.size() * 112*112*3);

  Mat img, face;
  for(int i = 0; i < imglist.size(); i++){
    img = imread(imglist[i], IMREAD_COLOR);

    face_detect(img, face);

    memcpy(faces+i*112*112*3, face.data, 112*112*3*sizeof(float));
  }

  return faces;
}


void face_detect(
