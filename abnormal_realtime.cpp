#include "opencv2/core/core.hpp"   
#include "opencv2/objdetect/objdetect.hpp"   
#include "opencv2/highgui/highgui.hpp"   
#include "opencv2/imgproc/imgproc.hpp"   
#include "opencv2/contrib/contrib.hpp"

#include <iostream>   
#include <stdio.h>   
#include <cv.h>  
#include <cxcore.h>  
#include <highgui.h>
#include <process.h>
#include <fstream>
#include "Windows.h"


using namespace std;
using namespace cv;

#define NUM_THREADS 2

string face_cascade_name = "haarcascade_frontalface_alt.xml";
string eye_cascade_name = "haarcascade_eye.xml";
string mouth_cascade_name = "haarcascade_mcs_mouth.xml";
//该文件存在于OpenCV安装目录下的\sources\data\haarcascades内，需要将该xml文件复制到当前工程目录下  

CascadeClassifier face_cascade;
CascadeClassifier eye_cascade;
CascadeClassifier mouth_cascade;
void detectAndDisplay(Mat frame);


int main(int argc, char** argv){
	if (!face_cascade.load(face_cascade_name)){
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
		return 0;
	}
	if (!eye_cascade.load(eye_cascade_name))
	{
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
		return 0;
	}
	if (!mouth_cascade.load(mouth_cascade_name))
	{
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
		return 0;
	}
	
	VideoCapture cap("rtsp://192.168.1.133:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif");
	if (!cap.isOpened())
	{
		return -1;
	}
	Mat image1; Mat image;
	bool stop = false;
	
	while (!stop)
	{
		cap >> image1;
		
			resize(image1, image, Size(400, 400), 0, 0, CV_INTER_LINEAR);
			imshow("当前视频", image);
			detectAndDisplay(image);
			if (waitKey(30) >= 0)
				stop = true;
		
	}
	return 0;
}

void detectAndDisplay(Mat face){
	std::vector<Rect> faces;
	Mat face_gray;
	Mat dst;
	Mat face1 = face.clone();
	//res = &IplImage(face1);//载入一张图片
	Rect rect;

	//人脸检测
	cvtColor(face, face_gray, CV_BGR2GRAY);//rgb类型转换为灰度类型
	equalizeHist(face_gray, face_gray);   //直方图均衡化  
	face_cascade.detectMultiScale(face_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1));
	if (faces.size() == 0){
		cout << "没有人脸" << endl;
		return;
	}

	for (int i = 0; i < faces.size(); i++){
		//Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);//画出人脸
		//ellipse(face, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 0), 2, 7, 0);
		rect.x = faces[i].x, rect.y = faces[i].y, rect.width = faces[i].width, rect.height = faces[i].height;
		//}
		dst = face(rect);//取出人脸图

		//口罩检测
		//直线检测
		Mat mtx=dst.clone();
		Mat mtx_gray, src;
	
			cvtColor(mtx, mtx_gray, CV_RGB2GRAY);
		
		Canny(mtx_gray, src, 50, 150, 3);//canny提取边缘信息
		IplImage *frame;
		frame = &IplImage(src);
		cvThreshold(frame, frame, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);//二值化
		Mat frame_img(frame);
		Mat midImage;
		midImage = frame_img(Range((rect.height) / 7 * 3, (rect.height) / 5 * 3), Range(0, rect.width));//裁取口罩部分
		equalizeHist(midImage, midImage);
		Mat dstImage;
		cvtColor(midImage, dstImage, CV_GRAY2BGR);
		vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
		HoughLinesP(midImage, lines, 1, CV_PI / 180, 46, 50, 10);	//直线检测
		/*for (size_t i = 0; i < lines.size(); i++)//画出直线
		{
			Vec4i l = lines[i];
			line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186, 88, 255), 1, CV_AA);
		}*/
		
		//嘴巴检测
		std::vector<Rect> mouths;
		Mat mes = dst.clone();
		cvtColor(mes, mes, CV_RGB2GRAY);
		equalizeHist(mes, mes);
		mes = mes(Range((rect.height) / 3*2, (rect.height)), Range(0, rect.width));//截取人脸下半部分

		mouth_cascade.detectMultiScale(mes, mouths, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		/*const static Scalar colors[] = { CV_RGB(0, 0, 255),//画出嘴巴
			CV_RGB(0, 128, 255),
			CV_RGB(0, 255, 255),
			CV_RGB(0, 255, 0),
			CV_RGB(255, 128, 0),
			CV_RGB(255, 255, 0),
			CV_RGB(255, 0, 0),
			CV_RGB(255, 0, 255) };
		int j = 0;	int radius = 0;
		for (vector<Rect>::const_iterator r = mouths.begin(); r != mouths.end(); r++, j++)
		{
			Point center;
			Scalar color = colors[j % 8];
			center.x = cvRound(r->x + r->width*0.5);
			center.y = cvRound(r->y + r->height*0.5);
			radius = (int)(cvRound(r->width + r->height)*0.25);
			circle(mes, center, radius, color, 3, 8, 0);
		}*/
		
		cout << "嘴巴个数："<<mouths.size() << endl;
		//口罩判断
		if (lines.size() != NULL || mouths.size() == 0)
		{
			cout << "第" << i + 1 << "个人" << "戴口罩" << endl;
		}
		else
			cout << "第" << i + 1 << "个人" << "不戴口罩" << endl;

		//墨镜检测
		//人脸yuv像素值检测
		Mat midimage = dst.clone();
		Mat input;
		Mat mask;
		float p = 0;
		IplImage *dst_image;
		mask = midimage(Range((rect.height) / 4, (rect.height) / 7 * 4), Range(0, rect.width));
		cvtColor(mask, input, CV_BGR2YCrCb);//rgb类型转换为ycrcb类型
		
		dst_image = &IplImage(input);//皮肤像素点检测统计
		int row = 0;
		int col = 0;
		int y = 0; int u = 0; int v = 0;
		for (row = 0; row < dst_image->height; row++){
			for (col = 0; col < dst_image->width; col++){
				y = CV_IMAGE_ELEM(dst_image, unsigned char, row, col * 3 + 0);
				u = CV_IMAGE_ELEM(dst_image, unsigned char, row, col * 3 + 1);
				v = CV_IMAGE_ELEM(dst_image, unsigned char, row, col * 3 + 2);
				if (133 <= u && u <= 173 && 77 <= v && v <= 127){
					p++;
				}
			}
		}
		float q = dst_image->height*dst_image->width;
		float t = p / q;
		cout << "皮肤占整体比重：" << t << endl;

		//人眼检测
		std::vector<Rect> eyes;
		Mat mask_gray;
		Mat endimage = dst.clone();
		endimage = endimage(Range(0, (rect.height) / 2), Range(0, rect.width));//截取眼睛部分
		cvtColor(endimage, mask_gray, CV_BGR2GRAY);
		equalizeHist(mask_gray, mask_gray);

		eye_cascade.detectMultiScale(mask_gray, eyes,1.1, 2, 0| CV_HAAR_SCALE_IMAGE,Size(30, 30));
		
		//int j1 = 0; int radius1 = 0;//画出眼睛
		//for (vector<Rect>::const_iterator nr = eyes.begin(); nr != eyes.end(); nr++)
		//{
		//	Point center;
		//	Scalar color = colors[j1 % 8];
		//	center.x = cvRound(nr->x + nr->width*0.5);
		//	center.y = cvRound(nr->y + nr->height*0.5);
		//	radius1 = cvRound((nr->width + nr->height)*0.25);
		//	circle(endimage, center, radius1, color, 3, 8, 0);//将眼睛也画出来，和对应人脸的图形是一样的  
		//}
		
		cout <<"眼睛个数："<< eyes.size() << endl;

		//墨镜判断
		if (t < 0.5 && eyes.size() == 0){
			cout << "第" << i + 1 << "个人" << "戴墨镜" << endl;
		}
		else
			cout << "第" << i + 1 << "个人" << "不戴墨镜" << endl;

	}
}