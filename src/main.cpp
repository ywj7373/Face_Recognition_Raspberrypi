#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

#include "utils.h"
#include "face_recognition.h"

using namespace std;
using namespace cv;

#define SKIP_FRAMES 2
#define DOWNSAMPLE_RATIO 4
#define MAX_NUMBER_OF_FACES 10

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
		
		int embedded_size = get_embedded_size(argv[1]);

		// Get Encodings
		vector<pair<string, vector<float>>> album_encodings;
		string filepath(argv[2]);
		ifstream input_file(filepath);
		if (!input_file.is_open()) {
			cerr << "Error loading encodings.txt\n";
				return -1;	
		}
		album_encodings = readEncodingsFromFile(input_file, embedded_size);
		
		// Open the default camera
		VideoCapture cap(0);	
		cap.set(CAP_PROP_FPS, 30); // Set FPS to 30	
		cv::Mat img, img_small, img_gray;
		int frame_cnt = 0;
		
		// Get Face Detector
		vector<string> faces_name(MAX_NUMBER_OF_FACES);
		CascadeClassifier face_cascade;
		string detector_path = "/home/pi/workspace/face_recognition_raspberrypi/face_detector/haarcascade_frontalface_default.xml";
		if (!face_cascade.load(detector_path)) {
			cerr << "Error loading xml\n";
			return -1;
		}
		
		// Bounding box
		int line_width = 3;
		std::vector<Rect> faces;
		
		timer_start(0);
		string fps = "";
		while(1) {
			// Measure FPS
			if (frame_cnt > 0 && frame_cnt % 30 == 0) {
				fps = "FPS: " + to_string(30 / timer_stop(0));
				timer_start(0);
			}
			
			// Grab a frame
			cap >> img;

			if (frame_cnt % SKIP_FRAMES == 0) {
				// Detect face
				resize(img, img_small, Size(), 1.0/DOWNSAMPLE_RATIO, 1.0/DOWNSAMPLE_RATIO);
				cvtColor(img_small, img_gray, COLOR_BGR2GRAY);
				equalizeHist(img_gray, img_gray);
				//timer_start(1);
				face_cascade.detectMultiScale(img_gray, faces, 1.1, 2, 9 | CASCADE_SCALE_IMAGE, Size(30, 30));
				//cout << "Face Detection Time: " << timer_stop(1) << "\n";
				
				// Face Recognition starts
				for(int i = 0; i < faces.size() && i < MAX_NUMBER_OF_FACES; i++) {
					vector<float> encoding;
					auto x = faces[i].x * DOWNSAMPLE_RATIO;
					auto y = faces[i].y * DOWNSAMPLE_RATIO;
					auto w = faces[i].width * DOWNSAMPLE_RATIO;
					auto h = faces[i].height * DOWNSAMPLE_RATIO;
					Mat face = img(Rect(x, y, w, h));
					
					// Encode image
					face = preprocess(face);
					timer_start(2);
					encoding = face_encoding(argv[1], face);
					cout << "Face Recognition Inference Time: " << timer_stop(2) << "\n";
					
					unordered_map<string, int> name_cnt;
					int max_cnt = 0;
					string max_num_name = "unidentified";
					// Compare with album encodings
					for(int j = 0; j < album_encodings.size(); j++) {
						auto img = album_encodings[j];
						string name = img.first;
						vector<float> img_encoding = img.second;
						
						bool less = match(img_encoding, encoding, 1.10f);
						if (less) {
							int cnt = ++name_cnt[name];
							if (cnt > max_cnt) {
								max_cnt = cnt;
								max_num_name = name;
							}
						}
					}
					cout << max_num_name << " identified!" << "\n";
					faces_name[i] = max_num_name;
				}
			}
			
			// Draw Bounding Box with name
			for(int i = 0; i < faces.size() && i < MAX_NUMBER_OF_FACES; i++) {
				Rect face = faces[i];
				auto x = face.x * DOWNSAMPLE_RATIO;
				auto y = face.y * DOWNSAMPLE_RATIO;
				auto width = (face.x + face.width) * DOWNSAMPLE_RATIO;
				auto height = (face.y + face.height) * DOWNSAMPLE_RATIO;
				cv::putText(img, faces_name[i], Point(x, y), 0, 0.75, Scalar(0, 255, 0), 2);
				cv::rectangle(img, Point(x, y+5), Point(width, height), Scalar(0, 255, 0), line_width); 
			}
			cv::putText(img,fps, Point(30, 30), 0, 0.75, Scalar(0, 255, 0), 2);
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

