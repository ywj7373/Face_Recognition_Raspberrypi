#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

void timer_start(int i);

double timer_stop(int i);

void writeEncodingsToFile(ofstream& output_file, vector<float>& data);

void writePersonNameToFile(ofstream& output_file, string name);

vector<pair<string, vector<float>>> readEncodingsFromFile(ifstream& input_file, int embedded_size);

string getFileName(string filepath);

string getPersonName(string filename);

Mat preprocess(Mat img);
