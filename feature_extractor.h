#ifndef _FEATURE_EXTRACTOR_DEFINED
#define _FEATURE_EXTRACTOR_DEFINED

#include <iostream>
#include <stdlib.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

using namespace std;
using namespace tflite;
using namespace cv;

class feature_extractor{
  private:
    unique_ptr<Interpreter> interpreter;
    int img_num;
    float* registered_feature;
    float threshold;

  public:
    feature_extractor(char* modelname, int n, float* imgs, float th);
    bool match(float* img);
    void get_feature(float* img, float* feature);
}

#endif
