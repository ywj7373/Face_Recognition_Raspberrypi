#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "face_recognition.h"
#include "utils.h"

using namespace std;
using namespace cv;

vector<Rect> face_detect(dlib::frontal_face_detector& detector, Mat img) {
	dlib::cv_image<dlib::bgr_pixel> cimg(img);
	vector<Rect> result;

	vector<dlib::rectangle> faces = detector(cimg);
	for(int i = 0; i < faces.size(); i++) {
		dlib::rectangle face = faces[i];
		result.push_back(Rect(face.left(), face.top(), face.width(), face.height()));
	}
	
	return result;
}

int main(int argc, char* argv[]) {	
	ofstream output_file("encodings.txt");
	
	// Get image directory
	if (argc < 2) {
		cerr << "Please pass album directory\n";
		return -1;
	}
	
	// Get model directory
	if (argc < 3) {
		cerr << "Please pass model directory\n";
		return -1;
	}
	
	// Get face detector
	dlib:: frontal_face_detector detector = dlib::get_frontal_face_detector();
	
	// Load images
	string img_path(argv[1]);	
	cv::String path(img_path + "/*.jpg");
	vector<cv::String> images;
	cv::glob(path, images, true);

	// Loop through album	
	for(int i = 0; i < images.size(); i++) {
		// Get image name
		string filename = getFileName(images[i]);
		string personName = getPersonName(filename);
		
		// Read image
		Mat img = imread(images[i]);
		
		if (!img.data) {
			cerr << "Could not open image!\n";
	       		continue;
		}	
		
		// Detect face from the image
		vector<Rect> faces = face_detect(detector, img);
		
		if (faces.size() == 0) {
			cout << "No face detected in " << filename << " image!\n";
			continue;
		}

		for(int j = 0; j < faces.size(); j++) {
			vector<float> encoding;
			Mat face = img(faces[j]);
			
			// Add the image name in the file
			writePersonNameToFile(output_file, personName);

			// Preprocess
			face = preprocess(face);
			
			// Encode image
			encoding = face_encoding(argv[2], face);
			cout << "Face of " << personName << " encoding completed\n";
			
			// Write it to encodings.txt
			writeEncodingsToFile(output_file, encoding);
			
			// Show face image
			imshow(personName, face);	
		}
	}	

	output_file.close();

	waitKey(0);
	return 0;
}
