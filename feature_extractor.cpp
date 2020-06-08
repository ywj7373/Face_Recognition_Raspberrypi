#include <iostream>
#include <stdlib.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include "feature_extractor.h"

using namespace std;
using namespace tflite;
using namespace cv;
using namespace dlib;

//you need to feed imgs as n*112*112*3 float array
feature_extractor::feature_extractor(char* modelname, int n, float* imgs, float th){
  //load model to interpreter
  unique_ptr<FlatBufferModel> model;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder(*model, resolver)(&interpreter);
  interpreter->AllocateTensors();
  threshold = th*th;

  //load registered imgs
  img_num = n;
  float* input, output;
  registered_feature = malloc(sizeof(float)*512*n);
  for(int i = 0; i < n; i++){
    input = interpreter->typed_input_tensor<float>(0);
    memcpy(input, imgs+112*112*3*i, 112*112*3*sizeof(float));
    interpreter->Invoke();
    output = interpreter->typed_output_tensor<float>(0);
    memcpy(registered_feature+512*i, output, 512*sizeof(float));
  }
}

void feature_extractor::get_feature(float* img, float* feature){
  float* input, output;
  input = interpreter->typed_input_tensor<float>(0);
  memcpy(input, img, 112*112*3*sizeof(float));
  interpreter->Invoke();
  output = interpreter->typed_output_tensor<float>(0);
  memcpy(feature, output, 512*sizeof(float));
}

bool feature_extractor::match(float* img){
  float* feature = malloc(512*sizeof(float));
  get_feature(img, feature);
  float dist;
  for(int i = 0; i < img_num; i++){
    dist = 0;
    for(int j = 0; j < 512; j++){
      dist += (registered_feature[512*i+j]-feature[j])
              *(registered_feature[512*i+j]-feature[j]);
    }
    if(dist < threshold){
      free(feature);
      return true;
    }
  }
  free(feature);
  return false;
}

feature_extractor::~feature_extractor(){
  free(registered_feature);
}








