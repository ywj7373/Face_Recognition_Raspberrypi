#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>

using namespace std;
using namespace cv;

#define SKIP_FRAMES 2
#define DOWNSAMPLE_RATIO 4

int main() {  
    try {
        cout << "OpenCV Version: " << CV_VERSION << endl;
   
        // Open the default camera
        VideoCapture cap(0);
	
	// Set FPS to 30
	cap.set(CAP_PROP_FPS, 30);
	
	cv::Mat img, img_small, img_gray;
	
	// Set Time
	time_t prevTime, curTime;
	prevTime = 0;
	
	// Get Face Detector
	CascadeClassifier face_cascade;
        if (!face_cascade.load("/home/pi/workspace/face_recognition_raspberrypi/haarcascade_frontalface_default.xml")) {
		cerr << "Error loading xml\n";
		return 0;
	}
	
	// Bounding box
	int line_width = 3;

	int count = 0;
	std::vector<Rect> faces;
	
	while(1) {
	    	// Calculate FPS
		if (count % 30 == 0) {
			time(&curTime);
	        	double diff = difftime(curTime, prevTime);
			cout << diff << endl;
			prevTime = curTime;
			double fps = 30/diff;
			cout << "Estimated FPS : " << fps << endl;
		}
		// Grab a frame
		cap >> img;
		
		// Resize image
		resize(img, img_small, Size(), 1.0/DOWNSAMPLE_RATIO, 1.0/DOWNSAMPLE_RATIO);
		// Change color to Gray
		cvtColor(img_small, img_gray, COLOR_BGR2GRAY);
		equalizeHist(img_gray, img_gray);

		// Detect Faces and skip frames
		if (count % SKIP_FRAMES == 0) {
			face_cascade.detectMultiScale(img_gray, faces, 1.1, 2, 9 | CASCADE_SCALE_IMAGE, Size(30, 30));
			cout << faces.size() << endl;
		}

		// Draw rectange
		for(int i = 0; i < faces.size(); i++) {
			Rect face = faces[i];
			cv::rectangle(img, Point(face.x*DOWNSAMPLE_RATIO, face.y*DOWNSAMPLE_RATIO), Point((face.x + face.width)*DOWNSAMPLE_RATIO, (face.y + face.height)*DOWNSAMPLE_RATIO), Scalar(0, 255, 0), line_width); 
		}

		imshow("result", img);
		
		int c = waitKey(1);
		// Press esc to quit
		if (c == 27) {
			break;
		}
		count++;
	}
        
    }
    catch (exception& e) {
        cout << e.what() << endl;
    }
}

