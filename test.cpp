#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdio.h>

using namespace std;

void get_img_list(const char* filename, vector<string> &imglist){
  ifstream f;
  f.open(filename);
  string buffer;
  while(f.peek() != EOF){
    getline(f, buffer);
    imglist.push_back(buffer);
  }
}

int main(){
  vector<string> ls;

  get_img_list("test.txt", ls);

  for(int i = 0; i < ls.size(); i++){
    cout << ls[i] << endl;
  }

  return 0;
}
