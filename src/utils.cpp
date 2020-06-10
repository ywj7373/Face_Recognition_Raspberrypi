#include "utils.h"

#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <utility>
#include <sys/time.h>

using namespace std;
using namespace cv;

#define IMAGE_WIDTH 112
#define IMAGE_HEIGHT 112

static struct timeval start_time[8];

void timer_start(int i) {
	gettimeofday(&start_time[i], NULL);
}

double timer_stop(int i) {
	struct timeval end_time;
	gettimeofday(&end_time, NULL);
	double time_taken = (end_time.tv_sec - start_time[i].tv_sec) * 1e6;
	time_taken = (time_taken + (end_time.tv_usec - start_time[i].tv_usec)) * 1e-6;
	return time_taken;
}

void writeEncodingsToFile(ofstream &output_file, vector<float> &data) {
	ostream_iterator<float> output_iterator(output_file, "\n");
	copy(data.begin(), data.end(), output_iterator);
}

void writePersonNameToFile(ofstream &output_file, string name) {
	output_file << name << "\n";
}

vector<pair<string, vector<float>>> readEncodingsFromFile(ifstream& input_file) {
	vector<pair<string, vector<float>>> encodings;
	string name;
	while (getline(input_file, name)) {
		vector<float> encoding;
		for(int i = 0; i < 256; i++) {
			string line;
			getline(input_file, line);
			encoding.push_back(stof(line));
		}
		encodings.push_back(make_pair(name, encoding));
	}
	
	return encodings;	
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
