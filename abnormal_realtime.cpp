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
//���ļ�������OpenCV��װĿ¼�µ�\sources\data\haarcascades�ڣ���Ҫ����xml�ļ����Ƶ���ǰ����Ŀ¼��  

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
			imshow("��ǰ��Ƶ", image);
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
	//res = &IplImage(face1);//����һ��ͼƬ
	Rect rect;

	//�������
	cvtColor(face, face_gray, CV_BGR2GRAY);//rgb����ת��Ϊ�Ҷ�����
	equalizeHist(face_gray, face_gray);   //ֱ��ͼ���⻯  
	face_cascade.detectMultiScale(face_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1));
	if (faces.size() == 0){
		cout << "û������" << endl;
		return;
	}

	for (int i = 0; i < faces.size(); i++){
		//Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);//��������
		//ellipse(face, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 0), 2, 7, 0);
		rect.x = faces[i].x, rect.y = faces[i].y, rect.width = faces[i].width, rect.height = faces[i].height;
		//}
		dst = face(rect);//ȡ������ͼ

		//���ּ��
		//ֱ�߼��
		Mat mtx=dst.clone();
		Mat mtx_gray, src;
	
			cvtColor(mtx, mtx_gray, CV_RGB2GRAY);
		
		Canny(mtx_gray, src, 50, 150, 3);//canny��ȡ��Ե��Ϣ
		IplImage *frame;
		frame = &IplImage(src);
		cvThreshold(frame, frame, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);//��ֵ��
		Mat frame_img(frame);
		Mat midImage;
		midImage = frame_img(Range((rect.height) / 7 * 3, (rect.height) / 5 * 3), Range(0, rect.width));//��ȡ���ֲ���
		equalizeHist(midImage, midImage);
		Mat dstImage;
		cvtColor(midImage, dstImage, CV_GRAY2BGR);
		vector<Vec4i> lines;//����һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������
		HoughLinesP(midImage, lines, 1, CV_PI / 180, 46, 50, 10);	//ֱ�߼��
		/*for (size_t i = 0; i < lines.size(); i++)//����ֱ��
		{
			Vec4i l = lines[i];
			line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186, 88, 255), 1, CV_AA);
		}*/
		
		//��ͼ��
		std::vector<Rect> mouths;
		Mat mes = dst.clone();
		cvtColor(mes, mes, CV_RGB2GRAY);
		equalizeHist(mes, mes);
		mes = mes(Range((rect.height) / 3*2, (rect.height)), Range(0, rect.width));//��ȡ�����°벿��

		mouth_cascade.detectMultiScale(mes, mouths, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		/*const static Scalar colors[] = { CV_RGB(0, 0, 255),//�������
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
		
		cout << "��͸�����"<<mouths.size() << endl;
		//�����ж�
		if (lines.size() != NULL || mouths.size() == 0)
		{
			cout << "��" << i + 1 << "����" << "������" << endl;
		}
		else
			cout << "��" << i + 1 << "����" << "��������" << endl;

		//ī�����
		//����yuv����ֵ���
		Mat midimage = dst.clone();
		Mat input;
		Mat mask;
		float p = 0;
		IplImage *dst_image;
		mask = midimage(Range((rect.height) / 4, (rect.height) / 7 * 4), Range(0, rect.width));
		cvtColor(mask, input, CV_BGR2YCrCb);//rgb����ת��Ϊycrcb����
		
		dst_image = &IplImage(input);//Ƥ�����ص���ͳ��
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
		cout << "Ƥ��ռ������أ�" << t << endl;

		//���ۼ��
		std::vector<Rect> eyes;
		Mat mask_gray;
		Mat endimage = dst.clone();
		endimage = endimage(Range(0, (rect.height) / 2), Range(0, rect.width));//��ȡ�۾�����
		cvtColor(endimage, mask_gray, CV_BGR2GRAY);
		equalizeHist(mask_gray, mask_gray);

		eye_cascade.detectMultiScale(mask_gray, eyes,1.1, 2, 0| CV_HAAR_SCALE_IMAGE,Size(30, 30));
		
		//int j1 = 0; int radius1 = 0;//�����۾�
		//for (vector<Rect>::const_iterator nr = eyes.begin(); nr != eyes.end(); nr++)
		//{
		//	Point center;
		//	Scalar color = colors[j1 % 8];
		//	center.x = cvRound(nr->x + nr->width*0.5);
		//	center.y = cvRound(nr->y + nr->height*0.5);
		//	radius1 = cvRound((nr->width + nr->height)*0.25);
		//	circle(endimage, center, radius1, color, 3, 8, 0);//���۾�Ҳ���������Ͷ�Ӧ������ͼ����һ����  
		//}
		
		cout <<"�۾�������"<< eyes.size() << endl;

		//ī���ж�
		if (t < 0.5 && eyes.size() == 0){
			cout << "��" << i + 1 << "����" << "��ī��" << endl;
		}
		else
			cout << "��" << i + 1 << "����" << "����ī��" << endl;

	}
}