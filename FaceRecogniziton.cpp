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
	//看一下样本的宽高
	int height = images[0].rows;
	int width = images[0].cols;
	cout << "imageHeight= " << height << endl;
	cout << "imageWidth= " << width << endl;
	//简单的交叉验证, 取前S-1个样本, 保留最后一个样本作为test set来检测model预测能力
	Mat testSample = images[int(images.size()) - 1];
	int testLabel = labels[labels.size() - 1];
	cout << "testLabel=" << testLabel << endl;
	namedWindow("testImage", WINDOW_AUTOSIZE);
	imshow("testImage", testSample);
	cv::waitKey(60);
	//弹出test 样本
	images.pop_back();
	labels.pop_back();
	//train it
	Ptr<BasicFaceRecognizer> model = EigenFaceRecognizer::create();
	model->train(images, labels);
	cout << "training OK" << endl;
	//看一下预测结果
	cout << "predictedLabel = " << model->predict(testSample) << endl;
	//create haarCascadeClaasifier
	CascadeClassifier faceDetector;
	faceDetector.load(haarFaceDataPath);

	//注意级联分类器是用来识别是不是人脸的, 而训练的model是用来确定是不是你

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
/* 这里的逻辑是: 先读取一帧图片, 
	然后利用haar级联分类器对该帧进行识别人脸,
	如果没有人脸, 就不必预测是谁
	直接等待10ms然后再读一帧, 等待时间可以修改*/
		flip(frame, frame, 1);
		faceDetector.detectMultiScale(frame, faces, 1.1, 3, 0, Size(50, 50), Size(800, 800));
		for (int i = 0; i < faces.size(); i++) {
		/* 这里即确定了是人脸, 但不知道是谁
		因此需要在这里进行特征脸识别
		即用训练好的model对该帧进行判别分类
		*/
			Mat roi = frame(faces[i]);    //提取感兴趣区域, area of interesting
			//每一个face[i]都是一个Rect类对象
			cout << "已提取感兴趣area" << endl;
			cvtColor(roi, dst, COLOR_BGR2GRAY); //用的EigenRec需要gray images
			resize(dst, dst, testSample.size());
			int label = 0;
			stringstream tempStream;
			tempStream << model->predict(dst);  //进行预测
			tempStream >> label;                            //stringstream流来类型转换
			cout << "predict label = " << label << endl; //输出预测结果
			rectangle(frame, faces[i], Scalar(0, 0, 255), 2, 8, 0);  //绘制矩形框
			/*显然这个解决方案并没有完全实现实时检测人脸并识别身份,
			这是因为绘制矩形框的时机选择了在predict之后
			可以思考绘制两个矩形框,一个识别人脸,一个辨识身份*/
			if (label == 1228) {
				stringstream convert;
				convert << label;
				string temp;
				convert >> temp;
				//把预测结果放到矩形框上
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
		cv::waitKey(1);
		char c = cv::waitKey(10);
		//按键q退出循环
		if (c == 27) {
			break;
		}
	}
	cv::waitKey(60);
	return 0;

}

