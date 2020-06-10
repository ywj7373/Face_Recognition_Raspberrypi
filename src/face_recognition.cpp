#include <iostream>
#include <cmath>
#include <vector>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/tools/gen_op_registration.h>
#include <tensorflow/lite/kernels/register.h>

#include "face_recognition.h"

using namespace std;
using namespace tflite;

#define IMAGE_WIDTH 112
#define IMAGE_HEIGHT 112

vector<float> face_encoding(char* modelPath, Mat img) {
	auto model = tflite::FlatBufferModel::BuildFromFile(modelPath);
	int embedded_size = 0;
	vector<float> encoding;
	
	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	unique_ptr<Interpreter> interpreter;
	InterpreterBuilder(*model.get(), resolver)(&interpreter);
	if (!interpreter) {
		cerr << "Failed to construct Interpreter\n";
	}	

	// Get model dimensions
	auto out = interpreter->outputs()[0];
	embedded_size = interpreter->tensor(out)->dims->data[1];
	
	// Start running model
	interpreter->AllocateTensors();	
	
	float* input = interpreter->typed_input_tensor<float>(0);
	memcpy(input, img.data, IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float));
	
	interpreter->Invoke();

	auto output = interpreter->tensor(interpreter->outputs()[0]);
	float* output_data = output->data.f;
	
	//l2_norm
	float l2 = 0.0f;
	for(int i = 0; i < embedded_size; i++) {
		l2 += pow(output_data[i], 2);
	}
	l2 = sqrt(l2);
	
	for(int i = 0; i < embedded_size; i++) {
		encoding.push_back(output_data[i]/l2);
	}

	return encoding;
}

bool match(vector<float> img1, vector<float> img2, float threshold) {
	float sum = 0.0f;

	for(int i = 0; i < img1.size(); i++) {
		sum += pow(img2[i]-img1[i], 2);	
	}
	
	if (sum <= threshold) {
		return true;
	}

	return false;
}

int get_embedded_size(char* modelPath) {
	auto model = tflite::FlatBufferModel::BuildFromFile(modelPath);
	int embedded_size = 0;
	vector<float> encoding;
	
	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	unique_ptr<Interpreter> interpreter;
	InterpreterBuilder(*model.get(), resolver)(&interpreter);
	if (!interpreter) {
		cerr << "Failed to construct Interpreter\n";
	}	

	// Get model dimensions
	auto out = interpreter->outputs()[0];
	embedded_size = interpreter->tensor(out)->dims->data[1];
	
	return embedded_size;
}

