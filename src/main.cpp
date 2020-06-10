#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <utility>

#include "utils.h"
#include "face_recognition.h"

using namespace std;
using namespace cv;

#define SKIP_FRAMES 2
#define DOWNSAMPLE_RATIO 4

int main(int argc, char* argv[]) {  
    try {
        cout << "OpenCV Version: " << CV_VERSION << endl;
   	
	// Get model directory
	if (argc < 2) {
		cerr << "Please pass model directory\n";
		return -1;
	}
	
	// Get encodings directory
	if (argc < 3) {
		cerr << "Please pass encodings.txt\n";
		return -1;
	}

	// Get Encodings
	vector<pair<string, vector<float>>> album_encodings;
	string filepath(argv[2]);
	ifstream input_file(filepath);
	if (!input_file.is_open()) {
		cerr << "Error loading encodings.txt\n";
	        return -1;	
	}
	album_encodings = readEncodingsFromFile(input_file);
	
        // Open the default camera
        VideoCapture cap(0);	
	cap.set(CAP_PROP_FPS, 30); // Set FPS to 30	
	cv::Mat img, img_small, img_gray;
	int frame_cnt = 0;
	
	// Get Face Detector
	CascadeClassifier face_cascade;
        string detector_path = "/home/pi/workspace/face_recognition_raspberrypi/face_detector/haarcascade_frontalface_default.xml";
	if (!face_cascade.load(detector_path)) {
		cerr << "Error loading xml\n";
		return -1;
	}
	
	// Bounding box
	int line_width = 3;
	std::vector<Rect> faces;
	
	while(1) {
		// Grab a frame
		cap >> img;
		
		// Preprocess image for face detection
		resize(img, img_small, Size(), 1.0/DOWNSAMPLE_RATIO, 1.0/DOWNSAMPLE_RATIO);
		cvtColor(img_small, img_gray, COLOR_BGR2GRAY);
		equalizeHist(img_gray, img_gray);

		// Face Recognition starts
		if (frame_cnt % SKIP_FRAMES == 0) {
			// Detect face
			face_cascade.detectMultiScale(img_gray, faces, 1.1, 2, 9 | CASCADE_SCALE_IMAGE, Size(30, 30));
				
			for(int i = 0; i < faces.size(); i++) {
				vector<float> encoding;
				auto x = faces[i].x * DOWNSAMPLE_RATIO;
				auto y = faces[i].y * DOWNSAMPLE_RATIO;
				auto w = faces[i].width * DOWNSAMPLE_RATIO;
				auto h = faces[i].height * DOWNSAMPLE_RATIO;
				Mat face = img(Rect(x, y, w, h));
				
				// Encode image
				face = preprocess(face);
				encoding = face_encoding(argv[1], face);
				
				// Compare with album encodings
				for(int j = 0; j < album_encodings.size(); j++) {
					auto img = album_encodings[j];
					string name = img.first;
					vector<float> img_encoding = img.second;
					
					bool less = match(img_encoding, encoding, 1.20f);
					if (less) {
						cout << name << " identified!!" << "\n";
						break;
					}
					else {
						cout << "Fail!!" << "\n";
					}
				}
			}
		}
		
		// Draw Bounding Box
		for(int i = 0; i < faces.size(); i++) {
			Rect face = faces[i];
			auto x = face.x * DOWNSAMPLE_RATIO;
			auto y = face.y * DOWNSAMPLE_RATIO;
			auto width = (face.x + face.width) * DOWNSAMPLE_RATIO;
			auto height = (face.y + face.height)*DOWNSAMPLE_RATIO;
			cv::rectangle(img, Point(x, y), Point(width, height), Scalar(0, 255, 0), line_width); 
		}
		imshow("result", img);
		
		// Press ESC to exit
		int c = waitKey(1);
		if (c == 27) {
			break;
		}

		frame_cnt++;
	}
        
    }
    catch (exception& e) {
        cout << e.what() << endl;
    }
}

