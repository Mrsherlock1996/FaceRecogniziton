#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::face;

string haarFaceDataPath = "C:/opencv/build/etc/haarcascades/haarcascade_frontalface_alt_tree.xml";

int main(int argc, char** argv)
{
	Mat testSample2 = cv::imread("C:/Users/xpp19/Desktop/test.jpg");
	if (testSample2.empty()) {
		cout << "no image..." << endl;
		return -1;
	}
	namedWindow("testImage2", WINDOW_AUTOSIZE);
	imshow("testImage2", testSample2);
	cv::waitKey(60);
	string fileName = "C:/Users/xpp19/Documents/test1228.txt";  //这里是我的脸的数据
	ifstream file(fileName.c_str(), ifstream::in);
	if (!file) {
		cout << "couble not load test.txt..." << endl;
		return -1;
	}
	string path, classLabel;
	vector<Mat> images;
	vector<int> labels;
	while (getline(file, path)) {
		getline(file, classLabel);
		int i = 0;
		//cout << "labels: " << labels[i] << endl;
		//cout << "path: " << path[i] << endl;
		i++;
		if (!path.empty() && !classLabel.empty()) {
			images.push_back(imread(path, IMREAD_GRAYSCALE));
			stringstream tempConvert;
			tempConvert << classLabel;
			int temp = 0;
			tempConvert >> temp;
			labels.push_back(temp);
			cout << "Convert Labels = " << temp << endl;
		}
	}
	//test image
	Mat testSample1 = images[1];
	namedWindow("testImage1", WINDOW_AUTOSIZE);
	imshow("testImage1", testSample1);
	cv::waitKey(60);
	int height = images[0].rows;
	int width = images[0].cols;
	cout << "imageHeight= " << height << endl;
	cout << "imageWidth= " << width << endl;
	Mat testSample = images[int(images.size()) - 1];
	int testLabel = labels[labels.size() - 1];
	cout << "testLabel=" << testLabel << endl;
	namedWindow("testImage", WINDOW_AUTOSIZE);
	imshow("testImage", testSample);
	cv::waitKey(60);
	//stringstream convertLabels;
	//string tempString;
	//convertLabels << testLabel;
	//convertLabels >> tempString;
	images.pop_back();
	labels.pop_back();
	//train it
	Ptr<BasicFaceRecognizer> model = EigenFaceRecognizer::create();
	model->train(images, labels);
	cout << "training OK" << endl;

	//recognition face
	/*int predictedLabel =*/
	//cout << "predictedLabel = " << model->predict(testSample) << endl;
	//create A cascadeClaasifier
	CascadeClassifier faceDetector;
	faceDetector.load(haarFaceDataPath);

	//Open camera
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		cout << "camera not open..." << endl;
		return -1;
	}
	Mat frame;
	string faceRecognitionWindow;
	namedWindow(faceRecognitionWindow, WINDOW_AUTOSIZE);
	vector<Rect> faces;
	Mat dst;
	cout << "进入识别区" << endl;
	int faceNumber = 0;
	while (capture.read(frame)) {
		flip(frame, frame, 1);
		faceDetector.detectMultiScale(frame, faces, 1.1, 3, 0, Size(50, 50), Size(800, 800));
		for (int i = 0; i < faces.size(); i++) {
			Mat roi = frame(faces[i]);    //提取感兴趣区域, area of interesting
			//每一个face[i]都是一个Rect类对象
			cout << "已提取感兴趣area" << endl;
			cvtColor(roi, dst, COLOR_BGR2GRAY);
			resize(dst, dst, testSample.size());
			int label = 0;
			stringstream tempStream;
			tempStream << model->predict(dst);
			tempStream >> label;
			cout << "predict label = " << label << endl; //预测我的脸
			//估计predict返回值给int类型时溢出了
			rectangle(frame, faces[i], Scalar(0, 0, 255), 2, 8, 0);
			if (label == 1228) {
				stringstream convert;
				convert << label;
				string temp;
				convert >> temp;
				putText(frame, string(temp), faces[i].tl(), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 2, 8);
				imwrite(format("C:/Users/xpp19/Documents/Myfaces/%d.jpg", faceNumber++), frame);
				cout << "已经写入第" << faceNumber << "个图片" << endl;
			}
			else {
				string temp = "fuck detector, this not me!";
				cout << temp << endl;
			}

		}
		cv::imshow(faceRecognitionWindow, frame);
		cv::waitKey(10);
		char c = cv::waitKey(10);
		//按键q退出循环
		if (c == 27) {
			break;
		}
	}
	cv::waitKey(60);
	return 0;

}

