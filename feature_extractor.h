#ifndef _FEATURE_EXTRACTOR_DEFINED
#define _FEATURE_EXTRACTOR_DEFINED

#include <iostream>
#include <stdlib.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

using namespace std;
using namespace tflite;

class feature_extractor{
  private:
    unique_ptr<Interpreter> interpreter;
    int img_num;
    float* registered_feature;
    float threshold;
    float* tmp;

  public:
    //input model's path and threshold
    feature_extractor(const char* modelname, float th);

    //input n and n*112*112*3 float arr
    void register_imgs(int n, float* imgs);

    //get either img contains register or not
    bool match(float* img);

    //get only feature
    void get_feature(float* img, float* feature);
}

#endif
