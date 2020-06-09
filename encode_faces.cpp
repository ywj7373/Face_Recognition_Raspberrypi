#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/tools/gen_op_registration.h>
#include <tensorflow/lite/kernels/register.h>

using namespace std;
using namespace cv;
using namespace tflite;

#define IMAGE_WIDTH 112
#define IMAGE_HEIGHT 112
#define EMBEDDED_SHAPE 256

void writeEncodingsToFile(ofstream &output_file, vector<vector<float>> &data) {
	for(int i = 0; i < data.size(); i++) {
		vector<float> encoding = data[i];
		ostream_iterator<float> output_iterator(output_file, "\n");
		copy(encoding.begin(), encoding.end(), output_iterator);
	}
}

void writePersonNameToFile(ofstream &output_file, string name) {
	output_file << name << "\n";
}

string getFileName(string filepath) {
	string filename = filepath;

	const size_t last_slash_idx = filename.find_last_of("\\/");
	if (string::npos != last_slash_idx) {
		filename.erase(0, last_slash_idx + 1);
	}

	const size_t period_idx = filename.rfind('.');
	if (string::npos != period_idx) {
		filename.erase(period_idx);
	}
	
	return filename;
}

Mat preprocess(Mat img) {
	Mat new_img;
	resize(img, new_img, Size(IMAGE_WIDTH, IMAGE_HEIGHT)); // resize
	new_img.convertTo(new_img, CV_32FC1); // convert to float
	new_img = new_img / 255.0f;
	return new_img;
}

int main(int argc, char* argv[]) {	
	vector<vector<float>> encodings;
	ofstream output_file("encodings.txt");

	if (argc < 2) {
		cerr << "Please pass album directory\n";
		return -1;
	}

	if (argc < 3) {
		cerr << "Please pass model directory\n";
		return -1;
	}
	
	string img_path(argv[1]);	
	cv::String path(img_path + "/*.jpg");
	vector<cv::String> images;
	cv::glob(path, images, true);

	// Load tflite model
	auto model = tflite::FlatBufferModel::BuildFromFile(argv[2]);
	
	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	unique_ptr<Interpreter> interpreter;
	InterpreterBuilder(*model.get(), resolver)(&interpreter);
	if (!interpreter) {
		cerr << "Failed to construct Interpreter\n";
		return -1;
	}
	
	// Print input and output dimensions	
	auto in = interpreter->inputs()[0];
	auto out = interpreter->outputs()[0];
	TfLiteIntArray* input_dims = interpreter->tensor(in)->dims;
	cout << "Input Dimension: " << input_dims->data[0] << " " \
		<< input_dims->data[1] << " " << input_dims->data[2] << " " << input_dims->data[3] << "\n";
	TfLiteIntArray* output_dims = interpreter->tensor(out)->dims;
	cout << "Output Dimension: " << output_dims->data[0] << " " << output_dims->data[1] << "\n";
	
	// Loop through album	
	for(int i = 0; i < images.size(); i++) {
		// Get image name
		string filename = getFileName(images[i]);
		
		// Read image
		Mat img = imread(images[i]);
		vector<float> encoding;

		if (!img.data) {
			cerr << "Could not open image!\n";
	       		continue;
		}	
		
		// Detect face from the image
		
		// Add the image name in the file
		writePersonNameToFile(output_file, filename);

		// Preprocess
		img = preprocess(img);

		// Encode image
		interpreter->AllocateTensors();
		float* input = interpreter->typed_input_tensor<float>(0);
		memcpy(input, img.data, IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float));

		interpreter->Invoke();

		auto output = interpreter->tensor(interpreter->outputs()[0]);
		float* output_data = output->data.f;
	
		for(int j = 0; j < EMBEDDED_SHAPE; j++) {
			encoding.push_back(output_data[j]);
		}

		cout << filename << " encoding completed\n";

		encodings.push_back(encoding);
	}	

	// Save 2D vector array in a file
	writeEncodingsToFile(output_file, encodings);
	output_file.close();

	cout << "Written to encodings.txt\n";

	return 0;
}
