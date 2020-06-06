
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>

using namespace dlib;
using namespace std;
using namespace cv;

#define SKIP_FRAMES 2
#define DOWNSAMPLE_RATIO 4

int main() {  
    try {
        cout << "OpenCV Version: " << CV_VERSION << endl;
   
        // Open the default camera
        VideoCapture cap(0);
	
	// Set FPS to 15
	cap.set(CAP_PROP_FPS, 15);
	
	cv::Mat img, img_small;
	time_t prevTime, curTime;
	prevTime = 0;
	
	// Get Face Detector
        frontal_face_detector detector = get_frontal_face_detector();
	
	// bounding box
	int line_width = 3;

	int count = 0;
	std::vector<dlib::rectangle> faces;
	
	while(1) {
	    	// Calculate FPS
		if (count % 120 == 0) {
			time(&curTime);
	        	double diff = difftime(curTime, prevTime);
			cout << diff << endl;
			prevTime = curTime;
			double fps = 120/diff;
			cout << "Estimated FPS : " << fps << endl;
		}
		// Grab a frame
		cap >> img;
		
		// Resize image
		resize(img, img_small, Size(), 1.0/DOWNSAMPLE_RATIO, 1.0/DOWNSAMPLE_RATIO);
		
		// Convert Mat to something dlib can deal with
		dlib::cv_image<bgr_pixel> cimg(img_small);

		// Detect Faces and skip frames
		//if (count % SKIP_FRAMES == 0)
		faces = detector(cimg);
		cout << faces.size() << endl;	
		// Draw rectange
		for(int i = 0; i < faces.size(); i++) {
			dlib::rectangle face = faces[i];
			cv::rectangle(img, Point(face.left()*DOWNSAMPLE_RATIO, \
			face.top()*DOWNSAMPLE_RATIO), Point(face.right()*DOWNSAMPLE_RATIO,\ 
			face.bottom()*DOWNSAMPLE_RATIO), Scalar(0, 255, 0), line_width); 
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

