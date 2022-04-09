// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <string.h>
#include <math.h>
#include <opencv2/imgproc/types_c.h>
#include "zmq.hpp"


 

using namespace cv;
using namespace std;

typedef unsigned char BYTE;
union Vision_detect_boundary
{
	struct {
		double distance;
		int orientation;
		double angle;
	}data;
	char buffer[sizeof(data) + 1 ];
};

union Vision_edge_extraction
{
	struct {
		double edge_k;
		double edge_b;
	}data;
	char buffer[sizeof(data) + 1];
};

#define pi 3.1415926
#define _T 0.00000001
#define _Max 10000
#define _Min 0
//#define _Leftwheel_outside  0.214286
#define _Leftwheel_topoutside  107
#define _Leftwheel_downoutside  35

#define _Rightwheel_topoutside 212
#define _Rightwheel_downoutside 282


float IOU(cv::Point p1, cv::Point p2)
{
	float m = p1.y - p1.x + p2.y - p2.x + _T;
	float left = min(p1.x, p2.x);
	float right = max(p1.y, p2.y);
	float c = right - left;
	return c / (m);
}
int overlap(Vec4i p, Vec4i q) {

	float kp = (p[3] - p[1]) / (p[2] - p[0] + _T);
	float bp = p[3] - kp * p[1];

	float kq = (q[3] - q[1]) / (q[2] - q[0] + _T);
	float bq = q[3] - kq * q[1];

	float x = (bq - bp) / (kp - kq);

	int pleft = p[0] - 10;
	int pright = p[2] + 10;

	if (x >= pleft && x <= pright)
		return 1;
	else
		return 0;
}

float calpararelined(Vec4i p, Vec4i q) {

	float kp = (p[3] - p[1]) / (p[2] - p[0] + _T);
	float bp = p[3] - kp * p[1];

	float kq = (q[3] - q[1]) / (q[2] - q[0] + _T);
	float kq2 = -kq;
	float bq2 = (q[3] + q[1]) / 2 - kq2 * (q[2] + q[0]) / 2;

	float x = (bq2 - bp) / (kp - kq2);
	float y = kq2 * x + bq2;

	float d = sqrtf(pow(y - (q[3] + q[1]) / 2, 2) + pow(x - (q[2] + q[0]) / 2, 2));

	return d;

}
Vec4i contact(Vec4i p1, Vec4i p2)
{
	Vec4i r;
	if (p1[0] > p1[2])
		p1 = Vec4i(p1[2], p1[3], p1[0], p1[1]);
	if (p2[0] > p2[2])
		p2 = Vec4i(p2[2], p2[3], p2[0], p2[1]);
	r[0] = min(p1[0], p2[0]);
	r[2] = max(p1[2], p2[2]);
	if (p1[0] < p2[0])
		r[1] = p1[1];
	else
		r[1] = p2[1];

	if (p1[2] < p2[2])
		r[3] = p2[3];
	else
		r[3] = p1[3];

	return r;

}

Vec4i hstack(Vec4i p1, Vec4i p2)
{
	Vec4i r;
	if (p1[1] > p1[3])
		p1 = Vec4i(p1[2], p1[3], p1[0], p1[1]);
	if (p2[1] > p2[3])
		p2 = Vec4i(p2[2], p2[3], p2[0], p2[1]);
	r[1] = min(p1[1], p2[1]);
	r[3] = max(p1[3], p2[3]);
	if (p1[1] < p2[1])
		r[0] = p1[0];
	else
		r[0] = p2[0];

	if (p1[3] < p2[3])
		r[2] = p2[2];
	else
		r[2] = p1[2];

	return r;

}



int findtailline(vector<Vec4i> &lines, cv::Point startpoint, cv::Point endpoint, vector<Vec4i> &filterlines) {

	vector<Vec4i>::iterator r;
	float dmin = _Max;
	int eraseloc = -1;
	Vec4i addpoint;
	string mode = "head";
	cv::Point nextstartpoint;
	cv::Point nextendpoint;

	for (r = lines.begin(); r != lines.end();)
	{



		float *dlist = (float*)malloc(4 * sizeof(float));
		dlist[0] = sqrtf(pow((startpoint.y - (*r)[1]), 2) + pow((startpoint.x - (*r)[0]), 2));
		dlist[1] = sqrtf(pow((startpoint.y - (*r)[3]), 2) + pow((startpoint.x - (*r)[2]), 2));
		dlist[2] = sqrtf(pow((endpoint.y - (*r)[1]), 2) + pow((endpoint.x - (*r)[0]), 2));
		dlist[3] = sqrtf(pow((endpoint.y - (*r)[3]), 2) + pow((endpoint.x - (*r)[2]), 2));

		int dmin_index = min_element(dlist, dlist + 4) - dlist;
		float d = *min_element(dlist, dlist + 4);

		free(dlist);
		if (d < dmin) {
			if (dmin_index % 2 == 0) {
				nextstartpoint = cv::Point((*r)[0], (*r)[1]);
				nextendpoint = cv::Point((*r)[2], (*r)[3]);

			}
			else {
				nextstartpoint = cv::Point((*r)[2], (*r)[3]);
				nextendpoint = cv::Point((*r)[0], (*r)[1]);

			}
			dmin = d;
			eraseloc = r - lines.begin();
			if (dmin_index >= 2)
				mode = "tail";
			else
				mode = "head";

		}

		r++;
		if (r == lines.end())
			if (mode == "tail")
				filterlines.push_back(Vec4i(nextstartpoint.x, nextstartpoint.y, nextendpoint.x, nextendpoint.y));
			else
				filterlines.insert(filterlines.begin(), Vec4i(nextstartpoint.x, nextstartpoint.y, nextendpoint.x, nextendpoint.y));




	}
	lines.erase(lines.begin() + eraseloc);
	if (mode == "head")
		return 1;
	else
		return 0;
	//if(eraseloc!=-1){
	//cv::Mat t=Mat::zeros(cv::Size(500,500),CV_8UC3);	
	//line(t, Point(lines[eraseloc][0],lines[eraseloc][1]), Point(lines[eraseloc][2],lines[eraseloc][3]), Scalar(0,125,120), 2, CV_AA);
	//line(t, laststartpoint, laststartpoint, Scalar(0,0,120), 3, CV_AA);	
	//cv::imshow("filter",t);
	//cv::waitKey(0);
	//}


}

void extractbox(const Mat &back,vector<vector<Point>> &contours,int &downbox,int k=1){

	int rows = back.rows;
	int cols = back.cols;
	
	
	vector<Vec4i> hierarcy;
	
	Mat dstImg;
	if(!k)
		dstImg = (back.clone()-255)*-1;
	else
		dstImg = back.clone();
	findContours(dstImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Rect> boundRects(contours.size());  
	vector<RotatedRect> boxs(contours.size()); 
	Point2f rect[4];
	int ymaxbox = _Min;
	if(!contours.size())
		return;

	for (int i = 0; i < contours.size(); i++)
	{
	 boxs[i] = minAreaRect(Mat(contours[i]));
	 boundRects[i] = boundingRect(Mat(contours[i]));
	 //circle(dstImg, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  
	 boxs[i].points(rect); 
	 //rectangle(back, Point(boundRects[i].x, boundRects[i].y), Point(boundRects[i].x + boundRects[i].width, boundRects[i].y + boundRects[i].height), Scalar(0, 255, 0), 2, 8);
	
	if(boundRects[i].height<rows/10 || boundRects[i].area()<rows*3 )//|| boundRects[i].height+boundRects[i].y==rows)
		continue;
	 
	//cv::Rect bottomRoi=cv::Rect(boundRect[i].x,boundRect[i].height-6,boundRect[i].width,5);
	
	//if(cv::sum(dst(bottomRoi))[0]<50)
		//continue;
	for (int j = 0; j < 4; j++)
	{
		//line(crop, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //绘制最小外接矩形每条边
		if (ymaxbox < rect[j].y)
		{
			ymaxbox = rect[j].y;
			downbox = i;
		}
	}
	
	}
	;
}

void extractSide(const Mat &back, Mat &side, vector<Point>&contour, int orie) {

	int rows = back.rows;
	int cols = back.cols;

	double base_k = 0;
	cv::Rect boundRect = boundingRect(Mat(contour));

	if (orie != 0)
		base_k = 1.0* back.rows / (back.cols - 15);
	vector<Point>::iterator p;
	

	for (p = contour.begin(); p != contour.end() ;)

	{

		if ((*p).y <= 3 || (*p).y >= rows - 3 || (*p).x <= 3 || (*p).x >= cols - 3) {
			p = contour.erase(p);
			continue;
		}
		
		if (orie == 1) {
			double k = (rows - (*p).y) / ((*p).x - 13 + _T);

			if (k <= base_k * 1.1 || k <= 0) {
				p = contour.erase(p);
				continue;
			}
		}
		if (orie == 2) {
			double k = (rows - (*p).y) / (cols - (*p).x - 13 + _T);
			if (k <= base_k * 1.1 || k <= 0) {
				p = contour.erase(p);
				continue;
			}
		}
		p++;

	}


	for (int i = 0; i < contour.size(); i++)
	{

		side.at<uchar>(contour[i].y, contour[i].x) = 255;
	}

}
void extractBackGround(Mat &dst,Mat & back){
	cv::Mat labels;
	
	int n_comps = connectedComponents(dst, labels, 4, CV_16U);
	//cout << n_comps << endl;
	
	//draw
	int rows = dst.rows;
	int cols = dst.cols;
	int *l = (int*)malloc(n_comps*sizeof(int));
	for (int i = 0; i < n_comps; i++) {
		*(l + i) = 0;
	}
	for (int x = 0; x < rows; x++)
	{
	 for (int y = 0; y < cols; y++)
		{
			int label = labels.at<short>(x, y);
			*(l + label) +=1;
			
		}
	}
	//for(int k=0;k<n_comps;k++)
	//	cout<<l+k<<*(l+k)<<endl;
	
	//cout<<max_element(l, l + n_comps)<<endl;
	int grasslabel=max_element(l+1, l + n_comps)-l;
	free(l);

	for (int x = 0; x < rows; x++)
	{
		for (int y = 0; y < cols; y++)
		{
			int label = labels.at<short>(x, y);
			if (label != grasslabel)
			{
				back.at<uchar>(x, y) = 0;
				
			}
			else
				back.at<uchar>(x, y) = 255;
			
		}
	}
	
	

	
	
}
void findLines(Mat back, vector<Vec4i> &filterlines)
{

	vector<Vec4i> plines;
	HoughLinesP(back, plines, 1, CV_PI / 180, 10, 10, 10);


	vector<Vec4i>::iterator p;
	if (plines.size() <= 1) {
		filterlines = plines;
		return;
	}
	float initK = 0;

	for (p = plines.begin(); p != plines.end() - 1;)
	{
		//cout<<"p0:"<<(*p)[0]<<"p2:"<<(*p)[2]<<endl;
		//cv::Point px1((*p)[0],(*p)[2]);
		//cv::Point py1(min((*p)[1],(*p)[3]),max((*p)[1],(*p)[3]));
		initK = ((*p)[3] - (*p)[1]) / ((*p)[2] - (*p)[0] + _T);
		vector<Vec4i>::iterator q;
		int flag = 0;
		for (q = p + 1; q != plines.end();)
		{
			cv::Point px1((*p)[0], (*p)[2]);
			cv::Point py1(min((*p)[1], (*p)[3]), max((*p)[1], (*p)[3]));

			cv::Point px2((*q)[0], (*q)[2]);
			cv::Point py2(min((*q)[1], (*q)[3]), max((*q)[1], (*q)[3]));
			cv::Mat t = Mat::zeros(cv::Size(500, 500), CV_8UC3);

			///*cout<<"p0:"<<(*p)[0]<<"p1:"<<(*p)[1]<<"p2:"<<(*p)[2]<<"p3:"<<(*p)[3]<<endl;
			//cout<<"q0:"<<(*q)[0]<<"q1:"<<(*q)[1]<<"q2:"<<(*q)[2]<<"q3:"<<(*q)[3]<<endl;
			//cout << "IOU(px1,px2)" << IOU(px1, px2) << endl;
			//cout << "IOU(py1,py2)" << IOU(py1, py2) <<endl;
			//line(t, Point((*p)[0],(*p)[1]), Point((*p)[2],(*p)[3]), Scalar(0,125,120), 2, CV_AA);
			//line(t, Point((*q)[0],(*q)[1]), Point((*q)[2],(*q)[3]), Scalar(0,0,120), 2, CV_AA);	
			//cv::imshow("filter",t);
			//cv::waitKey(0);*/

			float kp = ((*p)[3] - (*p)[1]) / ((*p)[2] - (*p)[0] + _T);
			float kq = ((*q)[3] - (*q)[1]) / ((*q)[2] - (*q)[0] + _T);
			//cv::Point mid1((px1.x+px1.y)/2,(py1.x+py1.y)/2);
			//cv::Point mid2((px2.x+px2.y)/2,(py2.x+py2.y)/2);
			//float mdim=sqrtf(pow((mid2.y-mid1.y),2)+pow((mid2.x-mid1.x),2));
			if ((IOU(px1, px2) <= 1 && IOU(py1, py2) <= 1)
				|| (IOU(px1, px2) <= 1.3 && IOU(py1, py2) <= 1.3 && abs(kp - kq) < 0.5 &&calpararelined((*p), (*q)) < 10)
				|| (IOU(px1, px2) <= 1.3&& abs(kp) < 0.1 &&abs(kq) < 0.1&&calpararelined((*p), (*q)) < 10)
				|| (IOU(py1, py2) <= 1.3&& abs(kp) > 10 && abs(kq) > 10 && calpararelined((*p), (*q)) < 10)
				)
			{
				if (abs(kp) > 10 && abs(kq) > 10)
					*p = hstack(*p, *q);
				else
					*p = contact(*p, *q);
				q = plines.erase(q);
				flag = 1;

			}
			//else if(overlap(*p,*q))
			//	q=plines.erase(q);
			else
				q++;
			if (q == plines.end() && flag != 1)
				p++;
		}

	}

	//find maxlen line
	float lemmax = _Min;
	int maxlennum = 0;
	for (int i = 0; i < plines.size(); i++) {
		//float k=(plines[i][3]-plines[i][1])/(plines[i][2]-plines[i][0]+_T);
		float d = sqrtf(pow((plines[i][3] - plines[i][1]), 2) + pow((plines[i][2] - plines[i][0]), 2));
		if (d > lemmax) {
			lemmax = d;
			maxlennum = i;
		}
		if (i == plines.size() - 1)
		{
			filterlines.push_back(plines[maxlennum]);
			plines.erase(plines.begin() + maxlennum);
		}
	}


	//filterlines.push_back(plines[0]);
	//plines.erase(plines.begin());
	//cv::Mat t=Mat::zeros(cv::Size(500,500),CV_8UC3);

	//sort lines
	int mode = 0;
	while (plines.size() != 0) {

		cv::Point lastendpoint = cv::Point(filterlines[filterlines.size() - 1][2], filterlines[filterlines.size() - 1][3]);

		cv::Point lasttstartpoint;
		if (mode == 0)
			lasttstartpoint = cv::Point(filterlines[0][0], filterlines[0][1]);
		else
			lasttstartpoint = cv::Point(filterlines[0][2], filterlines[0][3]);
		if (findtailline(plines, lasttstartpoint, lastendpoint, filterlines))
			mode = 1;


		//line(t, Point(filterlines[filterlines.size()-1][0],filterlines[filterlines.size()-1][1]), Point(filterlines[filterlines.size()-1][2],filterlines[filterlines.size()-1][3]), Scalar(0,125,120), 2, CV_AA);
		//line(t, Point(filterlines[0][0],filterlines[0][1]), Point(filterlines[0][2],filterlines[0][3]), Scalar(0,0,120), 2, CV_AA);	
		//cv::imshow("filter",t);
		//cv::waitKey(0);

	}

	// link all lines
	filterlines.reserve(filterlines.size() * 2 - 1);
	int start = 0;
	int p_idx = 0;
	int q_idx = 0;
	int dmin_index = 0;
	float d = 0;
	Mat t = Mat::zeros(back.size(), CV_8UC3);
	for (p = filterlines.begin(); p != filterlines.end();)
	{
		//cout << (*p)[0] << "::"<< (*p)[1] <<endl;

		vector<Vec4i>::iterator q = p + 1;

		/*line(t, Point((*p)[0], (*p)[1]), Point((*p)[2], (*p)[3]), Scalar(0, 0, 120), 2, CV_AA);
		line(t, Point((*q)[0], (*q)[1]), Point((*q)[2], (*q)[3]), Scalar(0,120, 0), 2, CV_AA);
		cv::imshow("filter", t);
		cv::waitKey(0);*/

		if (q == filterlines.end())
			break;
		//float kp = ((*p)[3] - (*p)[1]) / ((*p)[2] - (*p)[0] + _T);
		//float kq = ((*q)[3] - (*q)[1]) / ((*q)[2] - (*q)[0] + _T);
		// angp = atan(kp);
		// angq = atan(kq);
		//cout << angp << "::" << angq << endl;

		float *dlist = (float*)malloc(4 * sizeof(float));
		if (start == 0) {
			dlist[0] = sqrtf(pow(((*p)[1] - (*q)[1]), 2) + pow(((*p)[0] - (*q)[0]), 2));
			dlist[1] = sqrtf(pow(((*p)[1] - (*q)[3]), 2) + pow(((*p)[0] - (*q)[2]), 2));
			dlist[2] = sqrtf(pow(((*p)[3] - (*q)[1]), 2) + pow(((*p)[2] - (*q)[0]), 2));
			dlist[3] = sqrtf(pow(((*p)[3] - (*q)[3]), 2) + pow(((*p)[2] - (*q)[2]), 2));
			dmin_index = min_element(dlist, dlist + 4) - dlist;
			d = *min_element(dlist, dlist + 4);
			p_idx = int(dmin_index / 2) * 2;
			q_idx = dmin_index % 2 * 2;
			start = 1;
		}
		else {
			dlist[0] = sqrtf(pow(((*p)[3] - (*q)[1]), 2) + pow(((*p)[2] - (*q)[0]), 2));
			dlist[1] = sqrtf(pow(((*p)[3] - (*q)[3]), 2) + pow(((*p)[2] - (*q)[2]), 2));
			dmin_index = min_element(dlist, dlist + 2) - dlist;
			d = *min_element(dlist, dlist + 2);
			p_idx = 2;
			q_idx = dmin_index * 2;
		}



		/*if (q_idx == 2) {
			int q_idx2 = abs(q_idx - 2);

			(*q)[q_idx2] = (*q)[q_idx];
			(*q)[q_idx2 + 1] = (*q)[q_idx + 1];
			(*q)[q_idx] = tempx;
			(*q)[q_idx + 1] = tempy;
			q_idx = q_idx2;
		}*/
		if (d > 10) {

			filterlines.insert(q, Vec4i((*p)[p_idx], (*p)[p_idx + 1], (*q)[q_idx], (*q)[q_idx + 1]));
			p = p + 2;

			int q_idx2 = abs(q_idx - 2);
			int headx = (*p)[q_idx];
			int heady = (*p)[q_idx + 1];
			int tailx = (*p)[q_idx2];
			int taily = (*p)[q_idx2 + 1];

			(*p)[2] = tailx;
			(*p)[3] = taily;
			(*p)[0] = headx;
			(*p)[1] = heady;
		}
		else {
			float dq = sqrtf(pow(((*q)[3] - (*q)[1]), 2) + pow(((*q)[2] - (*q)[0]), 2));
			if (dq > 25) {
				(*p)[p_idx] = (*q)[q_idx];
				(*p)[p_idx + 1] = (*q)[q_idx + 1];
				p = p + 1;

				int q_idx2 = abs(q_idx - 2);
				int headx = (*p)[q_idx];
				int heady = (*p)[q_idx + 1];
				int tailx = (*p)[q_idx2];
				int taily = (*p)[q_idx2 + 1];

				(*p)[2] = tailx;
				(*p)[3] = taily;
				(*p)[0] = headx;
				(*p)[1] = heady;
			}

			else
			{
				int q_idx2 = abs(q_idx - 2);
				(*p)[p_idx] = (*q)[q_idx2];
				(*p)[p_idx + 1] = (*q)[q_idx2 + 1];
				q = filterlines.erase(q);

				int p_idx2 = abs(p_idx - 2);
				int headx = (*p)[p_idx2];
				int heady = (*p)[p_idx2 + 1];
				int tailx = (*p)[p_idx];
				int taily = (*p)[p_idx + 1];

				(*p)[2] = tailx;
				(*p)[3] = taily;
				(*p)[0] = headx;
				(*p)[1] = heady;


			}

		}
		free(dlist);


	}

	for (p = filterlines.begin(); p != filterlines.end() - 1;)
	{
		vector<Vec4i>::iterator q = p + 1;
		float kp = ((*p)[3] - (*p)[1]) / ((*p)[2] - (*p)[0] + _T);
		float kq = ((*q)[3] - (*q)[1]) / ((*q)[2] - (*q)[0] + _T);
		float angp = atan(kp);
		float angq = atan(kq);
		//cout << angp << "::" << angq << endl;
		if (min(double(abs(angp - angq)), pi - abs(angp - angq)) < pi / 9) {

			if (abs(kp) > 3 && abs(kq) > 3)
				*p = hstack(*p, *q);
			else
				*p = contact(*p, *q);
			q = filterlines.erase(q);


		}
		else
			if (q == filterlines.end())
				break;
			else
				p = q;
	}


}


void caledgeang(vector<Vec4i> &filterlines, float &ang, int mode, int orie) {
	int downmaxindex = -1;
	int dowmmax = _Min;
	//1 follow edge 0:random
	int min_l = 0, max_r = 0;
	if (mode) {
		vector<Vec4i>::iterator p;
		for (p = filterlines.begin(); p != filterlines.end();)
		{
			float kp = (-(*p)[2] + (*p)[0]) / ((*p)[3] - (*p)[1] + _T);

			if ((orie == 1 && kp < 0)
				|| (orie == 2 && kp > 0))
				p = filterlines.erase(p);
			else {
				p++;
			}

		}
	}

	for (int i = 0; i < filterlines.size(); i++) {
		if (dowmmax == max(filterlines[i][1], filterlines[i][3]))
		{

			float kp = (-filterlines[i][2] + filterlines[i][0]) / (filterlines[i][3] - filterlines[i][1] + _T);
			float kq = (-filterlines[downmaxindex][2] + filterlines[downmaxindex][0]) / (filterlines[downmaxindex][3] - filterlines[downmaxindex][1] + _T);
			if (kp*kq > 0 && abs(kp) > abs(kq)) {
				dowmmax = max(filterlines[i][1], filterlines[i][3]);
				downmaxindex = i;
			}
		}
		else if (dowmmax < max(filterlines[i][1], filterlines[i][3]))
		{

			dowmmax = max(filterlines[i][1], filterlines[i][3]);
			downmaxindex = i;

		}
		min_l = min(filterlines[i][0], filterlines[i][2]);
		max_r = max(filterlines[i][0], filterlines[i][2]);
	}
	if (downmaxindex == -1) {
		if (orie == 1)
			ang = pi / 2;
		else if (orie == 2)
			ang = -1 * pi / 2;
		else
			ang = 0;
		return;
	}
	float initK = (-filterlines[downmaxindex][2] + filterlines[downmaxindex][0]) / (filterlines[downmaxindex][3] - filterlines[downmaxindex][1] + _T);
	float init_angle = atan(initK);
	float len = sqrtf(pow((filterlines[downmaxindex][3] - filterlines[downmaxindex][1]), 2) + pow((filterlines[downmaxindex][2] - filterlines[downmaxindex][0]), 2));
	ang = init_angle;

	for (int i = downmaxindex + 1; i < filterlines.size(); i++) {

		if (max(filterlines[i][1], filterlines[i][3]) == max(filterlines[downmaxindex][1], filterlines[downmaxindex][3]))
			continue;
		float K = (-filterlines[i][2] + filterlines[i][0]) / (filterlines[i][3] - filterlines[i][1] + _T);
		float angle_i = atan(K);
		float d_i = sqrtf(pow((filterlines[i][3] - filterlines[i][1]), 2) + pow((filterlines[i][2] - filterlines[i][0]), 2));
		if (abs(angle_i - init_angle) > pi / 6)
			continue;
		else {
			len += d_i;
			ang = ang * (1 - d_i / len) + angle_i * d_i / len;
		}
		if (min(filterlines[i][0], filterlines[i][2]) == min_l)
			break;

		if (max(filterlines[i][0], filterlines[i][2]) == max_r)
			break;

	}
	if (downmaxindex >= 1)
		return;
	for (int i = downmaxindex - 1; i >= 0; i--) {

		if (max(filterlines[i][1], filterlines[i][3]) == max(filterlines[downmaxindex][1], filterlines[downmaxindex][3]))
			continue;
		float K = (-filterlines[i][2] + filterlines[i][0]) / (filterlines[i][3] - filterlines[i][1] + _T);
		float angle_i = atan(K);
		float d_i = sqrtf(pow((filterlines[i][3] - filterlines[i][1]), 2) + pow((filterlines[i][2] - filterlines[i][0]), 2));
		if (abs(angle_i - init_angle) > pi / 6)
			continue;
		else {
			len += d_i;
			ang = ang * (1 - d_i / len) + angle_i * d_i / len;
		}
		if (min(filterlines[i][0], filterlines[i][2]) == min_l)
			break;

		if (max(filterlines[i][0], filterlines[i][2]) == max_r)
			break;

	}
}

int caldownlocal(cv::Mat src,cv::Rect boundrect) {
	cv::Mat crop = src(boundrect);
	int cols = crop.cols;
	int rows = crop.rows;
	
	int height = 10;
	if (rows <= 10)
		height = rows;

	int width = 10;
	if (cols <= 10)
		width = cols;
	
	int step = static_cast<int>(cols / width);
	// << step << endl;
	int localx = cols;
	for (int i = 0; i < step; i++) {
		
		cv::Mat r = crop(cv::Rect(i*width, rows - height, width, height));
		//cout << cv::sum(r / 255)[0]<<endl;
		if (cv::sum(r / 255)[0] < height*width) {
			localx = i * width;
			break;
		}
	}
	if (localx + boundrect.x < src.cols / 2)
		return 2;
	else
		return 1;

}
void wrap(Mat &src, Mat &dst) {
	Point2f srcPoints[4];
	Point2f dstPoints[4];

	
	srcPoints[0] = Point2f(_Leftwheel_topoutside, 0);
	srcPoints[1] = Point2f(_Rightwheel_topoutside, 0);
	srcPoints[2] = Point2f(_Leftwheel_downoutside, src.rows - 1);
	srcPoints[3] = Point2f(_Rightwheel_downoutside, src.rows - 1);


	dstPoints[0] = Point2f(_Leftwheel_topoutside, 0);
	dstPoints[1] = Point2f(_Rightwheel_topoutside, 0);
	dstPoints[2] = Point2f(_Leftwheel_topoutside, src.rows - 1);
	dstPoints[3] = Point2f(_Rightwheel_topoutside, src.rows - 1);

	

	Mat M1 = getPerspectiveTransform(srcPoints, dstPoints);


	warpPerspective(src, dst, M1, cv::Size(src.cols, src.rows));



}



void CalboundaryInfo(Mat &src, Vision_detect_boundary &info)
{
	Mat dst,wrap_img;

	int width = src.cols;
	int height = src.rows;
	int orie = 0;
	float ang = 0;
	width = int(width / 2);
	height = int(height / 2);
	resize(src, src, Size(width, height));
	cv::cvtColor(src, dst, COLOR_BGR2GRAY);
	cv::threshold(dst, dst, 20, 255, THRESH_BINARY);//gray 70
	
	//adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, -2);
	int s = 5;
	Mat structureElement = getStructuringElement(MORPH_RECT, Size(s, s), Point(-1, -1));
	dilate(dst, dst, structureElement, Point(-1, -1), 1);
	
	wrap(dst, wrap_img);
	//wrap(src, src_wrap_img);
	//erode(dst, dst, structureElement, Point(-1, -1), 1);
	//morphologyEx(dst, dst, MORPH_OPEN, structureElement);
	cv::Mat fcrop;
	cv::Rect froi = cv::Rect(_Leftwheel_topoutside, 0, _Rightwheel_topoutside - _Leftwheel_topoutside, height);
	
	cv::Mat sideground = Mat::zeros(cv::Size(froi.width, froi.height), CV_8UC1);

	vector<vector<Point>> contours;
	Mat labelground = Mat::zeros(dst.size(), CV_8UC1);

	extractBackGround(wrap_img, labelground);

	fcrop = labelground(froi);

	int downbox = -1;
	vector<Point> contour;
	cv::Rect boundRect;
	vector<Vec4i> filterlines;

	extractbox(fcrop, contours, downbox, 0);
	if (contours.size() != 0 && downbox != -1) {


		contour = contours[downbox];
		boundRect = boundingRect(Mat(contour));
		extractSide(fcrop, sideground, contour, 0);
	}


	//distancefromedge(crop[0], boundRect[0]);
	//follow edge

	if (contour.size() == 0 || boundRect.area() < labelground.cols) {
		info.data.distance = height;
		info.data.orientation = 0;
		info.data.angle = 0;
	}
	else
	{
		info.data.distance = height - boundRect.y - boundRect.height;
		info.data.orientation = caldownlocal(fcrop, boundRect);
		if (boundRect.br().x < width / 2)
			orie = 1;
		else if (boundRect.tl().x > width / 2)
			orie = 2;
		else {
			orie = info.data.orientation;
		}


		if (boundRect.x > width / 3 && boundRect.x + boundRect.width < width * 2 / 3)
		{
			if (orie == 1)
				info.data.angle = pi / 2;
			else if (orie == 2)
				info.data.angle = pi / -2;
		}
		else {
			findLines(sideground, filterlines);
			caledgeang(filterlines, ang, 1, 0);
			if (orie == 0)
				info.data.angle = pi;
			else
				info.data.angle = ang;
		}
	}


}



void compute_info(Mat src,Vision_detect_boundary &detect_boundary) {

	
	
			
	//检测是否加成功
	if (!src.data)  
	{
		//cout << "Could not open or find the image" << endl;
		return;
	}
		
		

	CalboundaryInfo(src, detect_boundary);
	

		
	
	
}
int main(int argc,char *argv[])
{
	
	Vision_detect_boundary detect_boundary;
	Vision_edge_extraction edge_extraction;
	string zmqip = "tcp://127.0.0.1:10005";
	zmq::context_t ctx{ 1 };

	zmq::socket_t publisher(ctx, zmq::socket_type::pub);
	publisher.bind(zmqip);
	Mat src;
	//" net "
	compute_info(src,detect_boundary);

	zmq::message_t message;
	string topic = "detect_boundary";

	zmq::message_t message_topic(topic.data(), topic.size());

	publisher.send(message_topic, zmq::send_flags::sndmore);

	zmq::message_t message_buffer(sizeof(detect_boundary.buffer));

	memcpy(message_buffer.data(), detect_boundary.buffer, sizeof(detect_boundary.buffer));

	publisher.send(message_buffer, zmq::send_flags::none);
	return 0;
}


