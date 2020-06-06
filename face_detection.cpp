
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace dlib;
using namespace std;
using namespace cv;

int main() {  
    try {
        cout << "OpenCV Version: " << CV_VERSION << endl;
        Mat frame;
        
        // Open the default camera
        VideoCapture cap;
        cap.open(0)

        frontal_face_detector detector = get_frontal_face_detector();

        if (!cap.isOpened()) {
            cerr << "Error! Unable to open Camera" << endl;
            return -1;
        }

        for(;;) {
            cap.read(frame)
            if (frame.empty()) {
                cerr << "Error! Unable to read frame" << endl;
                break;
            }
            
            vector<dlib::rectangle> dets = detector(frame);
            cout << "Number of faces detected: " << dets.size() << endl;

            // Draw rectangle
            for(int i = 0; i < dets.size(); i++) {
                dlib::rectangle det = dets[i];
                rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width);
            }

            imshow("result", frame);
            int c = waitKey(1)

            // Press esc to quit
            if (c == 27) {
                break;
            }
        }
    }
    catch (exception& e) {
        cout << e.what() << endl;
    }

}

