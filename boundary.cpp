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
#pragma pack(1)
union Vision_mode
{
	struct
	{
		int mode;
	}data;
	char buffer[sizeof(data)];

};
union Vision_detect_boundary
{
	struct {
		double distance;
		int orientation;
		double angle;
	}data;
	char buffer[sizeof(data)];
};

union Vision_edge_extraction
{
	struct {
		double angle;
		double distance;
		int isside; //0 all grass ,1 r side ,2 left,3 no grass 
	}data;
	char buffer[sizeof(data)];
};
union Vision_inside
{
	struct {
		int area_ratio;

	}data;
	char buffer[sizeof(data)];
};
struct EdgeState
{
	float lastang;
	int lastside;
	int nosidecount;
	int side_advise;

};
#pragma pack(0)

struct EdgeState state = { 0,-1,10,0 };
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
	
	if(boundRects[i].height<rows/10 || boundRects[i].area()<rows*2 )//|| boundRects[i].height+boundRects[i].y==rows)
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

void extractSide(const Mat &back, Mat &side, vector<Point>&contour, int orie, int noside) {

	int rows = back.rows;
	int cols = back.cols;
	int maxy = _Min;
	double base_k = 0;
	int filterSide_count = 0;
	cv::Rect boundRect = boundingRect(Mat(contour));

	if (orie != 0)
		base_k = 1.0* back.rows / (back.cols - 15);
	vector<Point>::iterator p;


	for (p = contour.begin(); p != contour.end();)

	{

		if ((*p).y <= 3 || (*p).y >= rows - 3 || (*p).x <= 3 || (*p).x >= cols - 3) {
			p = contour.erase(p);
			continue;
		}

		if (orie == 1) {
			double k = (rows - (*p).y) / ((*p).x - 15 + _T);

			if (k <= base_k * 1.1&&k > 0) {
				maxy = max(maxy, (*p).y);
				p = contour.erase(p);

				filterSide_count++;
				continue;
			}
		}
		if (orie == 2) {
			double k = (rows - (*p).y) / (cols - (*p).x - 14 + _T);
			if (k <= base_k * 1.1&&k > 0) {
				maxy = max(maxy, (*p).y);
				p = contour.erase(p);
				filterSide_count++;
				continue;
			}
		}
		p++;

	}

	
	//if (maxy > rows / 2) {
	//	for (p = contour.begin(); p != contour.end();)
	//	{
	//		if ((*p).y >= maxy - maxy / 10)
	//			p = contour.erase(p);
	//		else
	//			p++;
	//	}
	//}
	if (orie != 0) {
		vector<vector<Point>>contours;
		vector<Point> temp;
		int empty = 0;
		int d = 0;
		int index = 0;
		for (int i = 0; i < rows; i++)
		{
			int flag = 0;
			for (p = contour.begin(); p != contour.end();)
			{
				if ((*p).y == i) {
					if (empty > 10) {
						if (temp.size() > 10) {
							contours.push_back(temp);
						}
						temp.clear();
						empty = 0;
						
						if (d > rows / 3)
							index = contours.size() - 1;
						d = 0;
					}
					temp.emplace_back((*p));
					p = contour.erase(p);

					flag++;
				}
				else
					p++;
			}
			if (flag == 0)
				empty++;
			else
				d++;
			if (i == rows - 1) {
				if (temp.size() > 10) {
					contours.push_back(temp);
					temp.clear();
				}
				
				if (d > rows / 3)
					index = contours.size() - 1;
				d = 0;
			}
		}
		contour.clear();
		if(contours.size()!=0)
			contour = contours[index];
	}
	if (filterSide_count > rows / 4)
		noside = 0;
	else
		noside = 1;

	for (int i = 0; i < contour.size(); i++)
	{

		side.at<uchar>(contour[i].y, contour[i].x) = 255;
	}
	/*cv::imshow("side", side);
	cv:; waitKey(0);*/
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
	int min_l = 0, max_r = 0,max_d=0,min_t=0;
	//cout << orie << endl;
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
			min_l = min(min_l,min(filterlines[i][0], filterlines[i][2]));
			max_r = max(max_r,max(filterlines[i][0], filterlines[i][2]));
			//min_t = min(min_t,min(filterlines[i][1], filterlines[i][3]));
			//max_d = max(max_d,max(filterlines[i][1], filterlines[i][3]));
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
			if (abs(angle_i - init_angle) > pi / 4 )
			{

				continue;
			}

			else {
				len += d_i;
				ang = ang * (1 - d_i / len) + angle_i * d_i / len;
			}
			
			if (min(filterlines[i][0], filterlines[i][2]) == min_l)
					break;

			if (max(filterlines[i][0], filterlines[i][2]) == max_r)
					break;
			
			/*else
			{
				if (min(filterlines[i][1], filterlines[i][3]) == min_t)
					break;

				if (max(filterlines[i][1], filterlines[i][3]) == max_d)
					break;
			}*/

		}
		if (downmaxindex >= 1)
			return;
		for (int i = downmaxindex - 1; i >= 0; i--) {

			if (max(filterlines[i][1], filterlines[i][3]) == max(filterlines[downmaxindex][1], filterlines[downmaxindex][3]))
				continue;
			float K = (-filterlines[i][2] + filterlines[i][0]) / (filterlines[i][3] - filterlines[i][1] + _T);
			float angle_i = atan(K);
			float d_i = sqrtf(pow((filterlines[i][3] - filterlines[i][1]), 2) + pow((filterlines[i][2] - filterlines[i][0]), 2));
			if (abs(angle_i - init_angle) > pi / 4)
				continue;
			else {
				len += d_i;
				ang = ang * (1 - d_i / len) + angle_i * d_i / len;
			}
			
			if (min(filterlines[i][0], filterlines[i][2]) == min_l)
				break;

			if (max(filterlines[i][0], filterlines[i][2]) == max_r)
				break;
			
			/*else
			{
				if (min(filterlines[i][1], filterlines[i][3]) == min_t)
					break;

				if (max(filterlines[i][1], filterlines[i][3]) == max_d)
					break;
			}*/

		}
	}
	else
	{
		std::vector<std::pair<float, float>> k_list;
		if (filterlines.size() > 1) {
			for (int i = 0; i < filterlines.size() - 1; ) {
				float ki = (-filterlines[i][2] + filterlines[i][0]) / (filterlines[i][3] - filterlines[i][1] + _T);
				float d_i = sqrtf(pow((filterlines[i][3] - filterlines[i][1]), 2) + pow((filterlines[i][2] - filterlines[i][0]), 2));
				float angi = atan(ki);
				//cout << "angi::" << angi * 180 / pi << endl;
				for (int j = i + 1; j < filterlines.size(); j++) {
					float kj = (-filterlines[j][2] + filterlines[j][0]) / (filterlines[j][3] - filterlines[j][1] + _T);
					float d_j = sqrtf(pow((filterlines[j][3] - filterlines[j][1]), 2) + pow((filterlines[j][2] - filterlines[j][0]), 2));
					float angj = atan(kj);
					//cout << "angj::" << angj * 180 / pi << endl;

					if (abs(atan(ki) - angj) < pi /4) {
						d_i += d_j;
						angi = angi * (1 - d_j / d_i) + angj * d_j / d_i;
					}
					else
					{
						if (d_i > 15) {
							//k_list.push_back(std::make_pair(d_i, angi));
							std::pair<float, float> ppair = std::make_pair(d_i, angi);
							if (k_list.size() == 0)
								k_list.push_back(ppair);
							else {
								if (abs(ppair.second - (*(k_list.end() - 1)).second) < pi / 6)
								{
									(*(k_list.end() - 1)).first += ppair.first;
									(*(k_list.end() - 1)).second = (1 - ppair.first) / (*(k_list.end() - 1)).first*(*(k_list.end() - 1)).second + ppair.first / (*(k_list.end() - 1)).first*ppair.second;
								}
								else
								{
									k_list.emplace_back(ppair);
								}

							}
						}
						if (j == filterlines.size() - 1) {
							k_list.push_back(std::make_pair(d_j, angj));
						}
						i += j;
						break;
					}
					if (j == filterlines.size() - 1) {
						k_list.push_back(std::make_pair(d_i, angi));
						i = j;
					}
				}

			}
		}
		else
		{
			float ki = (-filterlines[0][2] + filterlines[0][0]) / (filterlines[0][3] - filterlines[0][1] + _T);
			float d_i = sqrtf(pow((filterlines[0][3] - filterlines[0][1]), 2) + pow((filterlines[0][2] - filterlines[0][0]), 2));
			k_list.push_back(std::make_pair(d_i, atan(ki)));
		}
		int maxlen = _Min;
		if (orie == 2)
		{
			//cout << "::" << k_list.size() << endl;
			for (int s = 0; s < k_list.size(); s++) {
				if (k_list[s].second <- pi /6) {
					if (k_list[s].first > maxlen) {
						maxlen = k_list[s].first;
						ang = k_list[s].second;
					}
				}

			}
		}
		else if (orie ==1)
		{
			for (int s = 0; s < k_list.size(); s++) {
				if (k_list[s].second > pi /6) {
					if (k_list[s].first > maxlen) {
						maxlen = k_list[s].first;
						ang = k_list[s].second;
					}
				}

			}
		}
		else {
			for (int s = 0; s < k_list.size(); s++) {
				if (abs(k_list[s].second) < pi / 4) {
					if (k_list[s].first > maxlen) {
						maxlen = k_list[s].first;
						ang = k_list[s].second;
					}
				}

			}
		}
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

void Isside(Mat src, int& inside) {
	Mat dst, wrap_img, src_wrap_img;

	int width = src.cols;
	int height = src.rows;

	width = int(width / 2);
	height = int(height / 2);
	resize(src, src, Size(width, height));
	cv::cvtColor(src, dst, COLOR_BGR2GRAY);
	cv::threshold(dst, dst, 20, 255, THRESH_BINARY);//gray 70

	wrap(dst, wrap_img);

	//erode(dst, dst, structureElement, Point(-1, -1), 1);
	//morphologyEx(dst, dst, MORPH_OPEN, structureElement);
	cv::Rect froi = cv::Rect(_Leftwheel_topoutside, 0, _Rightwheel_topoutside - _Leftwheel_topoutside, height);
	cv::Mat fcrop = wrap_img(froi);

	double area_ratio = cv::sum(fcrop / 255)[0] / (fcrop.cols*fcrop.rows);

	if (area_ratio > 0.95)
		inside = 0;
	else if (area_ratio < 0.1)
		inside = 2;
	else
		inside = 1;
}

double convert2realdistance(int d)
{
	int c = static_cast<int>(d / 30);
	int lower = 0, up = 0;
	double real_distance = 0;
	switch (c)
	{
	case 0:
		lower = 35;
		up = 60;
		break;
	case 1:
		lower = 22.6;
		up = 35;
		break;
	case 2:
		lower = 15.5;
		up =22.6;
		break;
	case 4:
		lower = 10;
		up = 15.5;
		break;
	case 5:
		lower = 5.2;
		up = 15.5;
		break;
	case 6:
		lower = 0;
		up = 5.2;
		break;
	default:
		break;
	}
	real_distance = up - (up - lower)*(d - c * 30) / 30.0;

	return real_distance;
}

void CalboundaryInfo(Mat &src, Vision_detect_boundary &info)
{
	Mat dst,wrap_img, src_wrap_img;

	int width = src.cols;
	int height = src.rows;
	int orie = 0;
	float ang = 0;
	int noside = 0;
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
		
	}


	//distancefromedge(crop[0], boundRect[0]);
	//follow edge

	if (contour.size() == 0 || boundRect.area() < labelground.cols) {
		info.data.distance = 60;
		info.data.orientation = 0;
		info.data.angle = 0;
	}
	else
	{
		
		info.data.distance = convert2realdistance(boundRect.y + boundRect.height);
		if (info.data.distance ==0 && cv::sum(fcrop(cv::Rect(0, height - 1, _Rightwheel_topoutside - _Leftwheel_topoutside, 1)) / 255)[0] > 0)
		{
			info.data.distance = 1;
		}
		info.data.orientation = caldownlocal(fcrop, boundRect);
		if (boundRect.br().x < fcrop.cols / 2)
			orie = 1;
		else if (boundRect.tl().x > fcrop.cols / 2)
			orie = 2;
		else {
			orie = info.data.orientation;
		}


		if (boundRect.x > fcrop.cols / 3 && boundRect.x + boundRect.width < fcrop.cols * 2 / 3)
		{
			if (orie == 1)
				info.data.angle = 0;// pi / 2;
			else if (orie == 2)
				info.data.angle = pi;// pi / -2;
		}
		else {
			extractSide(fcrop, sideground, contour, 0, noside);
			findLines(sideground, filterlines);
			caledgeang(filterlines, ang, 1, orie);
			if (orie == 0)
				info.data.angle = 0;
			else {
				//info.data.angle = ang;
				if (ang > 0)
					info.data.angle = pi - ang;
				else if (ang < 0)
					info.data.angle = abs(ang);
				else
					info.data.angle = 0;
			}

		}
	}
	/*rectangle(src_wrap_img(froi), Point(boundRect.x, boundRect.y), Point(boundRect.x + boundRect.width, boundRect.y + boundRect.height), Scalar(255, 0, 0), 2, 8);
	cv::imshow("t", src_wrap_img);
	cv::waitKey(0);*/
}
int comparemidside(cv::Rect backRect[], vector<Point> contour[], vector<cv::Mat>crop, int noside[], int siderows, int sidecols)
{
	int orei = 0;


	int sl = cv::sum(crop[2] / 255)[0];
	int sr = cv::sum(crop[1] / 255)[0];

	if (contour[1].size() != 0 && backRect[1].x <= 5 && !(noside[1] == 1 && backRect[1].y > 5))
	{
		if (sr >= sl)
			orei = 1;
		else if (contour[2].size() != 0 && backRect[2].br().x >= sidecols - 5 && !(noside[2] == 1 && backRect[2].y > 5))
			orei = 2;

	}
	else {
		if (sl >= sr && contour[2].size() != 0 && backRect[2].br().x >= sidecols - 5 && !(noside[2] == 1 && backRect[2].y > 5))
			orei = 2;
	}

	return orei;
}
int computesidetype(vector<Point> contour, cv::Mat side_img, int sidenum)
{
	int rows = side_img.rows;
	int cols = side_img.cols;
	cv::Point left(_Max, _Min), top(_Max, _Min), down(_Max, _Min), right(_Max, _Min);
	int  leftflag = 0, topflag = 0, downflag = 0, rightflag = 0;

	double	base_k = 1.0* side_img.rows / (side_img.cols - 15);
	double	base_k2 = 1.0* side_img.rows / (side_img.cols - 25);

	if (sidenum == 1) {

		for (int i = 0; i < contour.size(); i++) {
			if (contour[i].x < 5 && contour[i].y>5 && contour[i].y < rows - 5)
			{
				left.x = min(left.x, contour[i].y);
				left.y = max(left.y, contour[i].y);
				leftflag = 1;
			}

			if (contour[i].x > 5 && contour[i].x < cols - 5 && contour[i].y < 5)
			{
				top.x = min(top.x, contour[i].x);
				top.y = max(top.y, contour[i].x);
				topflag = 1;
			}

			if (contour[i].x > 5 && contour[i].x < 10 && contour[i].y > rows - 5)
			{
				down.x = min(down.x, contour[i].x);
				down.y = max(down.y, contour[i].x);
				downflag = 1;
			}

			if (contour[i].x > 10 && contour[i].y > 5 && contour[i].y < rows - 5)
			{

				double k = (rows - contour[i].y) / (contour[i].x - 15 + _T);
				if (k >= base_k && k <= base_k2)
				{
					right.x = min(right.x, contour[i].y);
					right.y = max(right.y, contour[i].y);
				}
				rightflag = 1;
			}
		}

		/*if ((topflag == 1 && downflag == 1)
			|| (topflag == 1 && rightflag == 1)
			|| (rightflag == 1 && downflag == 1)
			)
			return 1;
		else
			return 0;*/
		//cout << "cc:" << top.y - top.x << "::" << topflag << endl;
		//cout << "cc2:" << left.y - left.x << "::" << leftflag << endl;
		if ((top.y - top.x > 20)
			|| (left.y - left.x > 30))
			return 0;
		else
			return 1;


	}
	else {

		for (int i = 0; i < contour.size(); i++) {
			if (contour[i].x > cols - 5 && contour[i].y > 5 && contour[i].y < rows - 5)
			{
				right.x = min(right.x, contour[i].y);
				right.y = max(right.y, contour[i].y);
				rightflag = 1;
			}

			if (contour[i].x > 5 && contour[i].x < cols - 5 && contour[i].y < 5)
			{
				top.x = min(top.x, contour[i].x);
				top.y = max(top.y, contour[i].x);
				topflag = 1;
			}

			if (contour[i].x > 5 && contour[i].x < 10 && contour[i].y > rows - 5)
			{
				down.x = min(down.x, contour[i].x);
				down.y = max(down.y, contour[i].x);
				downflag = 1;
			}

			if (contour[i].x < cols - 10 && contour[i].y > 5 && contour[i].y < rows - 5)
			{

				double k = (rows - contour[i].y) / (cols - contour[i].x - 14 + _T);
				if (k >= base_k && k <= base_k2)
				{
					left.x = min(left.x, contour[i].y);
					left.y = max(left.y, contour[i].y);
				}
				leftflag = 1;
			}
		}

		/*if ((topflag == 1 && downflag == 1)
			|| (topflag == 1 && leftflag == 1)
			|| (leftflag == 1 && downflag == 1)
			)
			return 1;
		else
			return 0;*/

		if ((top.y - top.x > 20)
			|| (right.y - right.x > 30))
			return 0;
		else
			return 1;


	}



}
void CaledgeInfo(Mat &src, int &isside, float &lastang, int &distance,int& inside,int side_advise)
{
	Mat dst, wrap_img, src_wrap_img;
	float ang[3] = { 0 };
	int noside[3] = { 0 };

	int orie = 0;
	int width = src.cols;
	int height = src.rows;
	/*int turnline = static_cast<int>(height * 2 / 3);
	int topline = static_cast<int>(height / 6);
	int downline = static_cast<int>(height * 5 / 6);*/

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
	cv::Mat fcrop, rcrop, lcrop;
	cv::Rect froi, lroi, rroi, lwheelroi, rwheelroi;
	froi = cv::Rect(_Leftwheel_topoutside, 0, _Rightwheel_topoutside - _Leftwheel_topoutside, height);
	lroi = cv::Rect(0, 0, _Leftwheel_topoutside - 1, height);
	rroi = cv::Rect(_Rightwheel_topoutside, 0, width - _Rightwheel_topoutside, height);

	vector<cv::Rect>roi = { froi,rroi,lroi };

	vector<vector<Point>> contours[3];
	vector<Vec4i> filterlines[3];
	Mat labelground = Mat::zeros(dst.size(), CV_8UC1);
	Mat fsideground = Mat::zeros(cv::Size(froi.width, froi.height), CV_8UC1);
	Mat rsideground = Mat::zeros(cv::Size(rroi.width, rroi.height), CV_8UC1);
	Mat lsideground = Mat::zeros(cv::Size(lroi.width, lroi.height), CV_8UC1);
	vector<cv::Mat>sideground = { fsideground,rsideground,lsideground };

	extractBackGround(wrap_img, labelground);

	fcrop = labelground(froi);
	rcrop = labelground(rroi);
	lcrop = labelground(lroi);

	

	vector<cv::Mat>crop = { fcrop,rcrop,lcrop };
	int downbox[3] = { -1,-1,-1 };//0:f 1:r 2:l
	vector<Point> contour[3];

	cv::Rect boundRect[3];

	for (int k = 0; k < 3; k++) {
		extractbox(crop[k], contours[k], downbox[k], k);
		if (contours[k].size() == 0 || downbox[k] == -1)
			continue;

		contour[k] = contours[k][downbox[k]];
		boundRect[k] = boundingRect(Mat(contour[k]));
		extractSide(crop[k], sideground[k], contour[k], k, noside[k]);

	}

	//follow edge
	//cv::imwrite("l.jpg", wrap_img);
	int rroi_nograssarea = 0, lroi_nograssarea = 0;
	if (contour[0].size() != 0 || contour[1].size() != 0 || contour[2].size() != 0)
	{

		if (contour[0].size() == 0 || boundRect[0].area() < labelground.cols || boundRect[0].br().y <= height / 2) {

			int sidetype1 = 1, sidetype2 = 1;   // 1 grass 2 noside or norotation 3 side

			if (contour[1].size() > 10) {
				
				int right_linenograsssum =3* int(boundRect[0].height / 2) - cv::sum(crop[1](cv::Rect(0, boundRect[1].y,3,int(boundRect[1].height /2))) / 255)[0];
				
				if (noside[1] == 1 || boundRect[1].br().y < crop[1].rows / 2 || (boundRect[1].y>crop[1].rows / 10 && contour[0].size() == 0) || right_linenograsssum>50) {

					sidetype1 = 2;
				}
				else
				{
					sidetype1 = 3;

				}

			}

			if (contour[2].size() > 10) {
				int left_linenograsssum = 3 * int(boundRect[2].height / 2) - cv::sum(crop[2](cv::Rect(crop[2].cols-3, boundRect[2].y, 3, int(boundRect[2].height / 2))) / 255)[0];
				
				if (noside[2] == 1 || boundRect[2].br().y < crop[2].rows / 2 || (boundRect[1].y>crop[2].rows / 10 && contour[0].size() == 0) || left_linenograsssum > 50) {

					sidetype2 = 2;
				}
				else
				{
					sidetype2 = 3;

				}

			}

			int choose = sidetype2 + sidetype1 * 10;
			int need_compute_lineangel = 0;
			switch (choose)
			{
			case 11:
				isside = -1;
				lastang = 0;

				break;
			case 12:
				isside = 2;
				lastang = 0;
				break;
			case 13:
				isside = 2;

				need_compute_lineangel = computesidetype(contour[2], crop[2], 2);

				break;
			case 21:
			case 22:
			case 23:
				isside = 1;
				lastang = 0;
				break;
			case 31:
			case 32:
			case 33:
				if (side_advise == 0) {
					rroi_nograssarea = rroi.width*rroi.height - cv::sum(dst(rroi) / 255)[0];

					lroi_nograssarea = lroi.width*lroi.height - cv::sum(dst(lroi) / 255)[0];

					if (rroi_nograssarea > rroi.width * 5) {
						isside = 1;
						need_compute_lineangel = computesidetype(contour[1], crop[1], 1);
					}
					else if (lroi_nograssarea > lroi.width * 5 && sidetype2 == 3)
					{
						isside = 2;
						need_compute_lineangel = computesidetype(contour[2], crop[2], 2);
					}
					else
					{
						isside = -1;
						need_compute_lineangel = 0;
					}
				}
				else
				{
					isside = side_advise;
					need_compute_lineangel = computesidetype(contour[side_advise], crop[side_advise], side_advise);
				}

				break;
			default:
				break;
			}

			if (need_compute_lineangel) {
				findLines(sideground[isside], filterlines[isside]);
				if (filterlines[isside].size() != 0) {
					if (contour[0].size() != 0 && boundRect[0].br().y>height/3)
						if (isside == 1)
							orie = 2;
						else if (isside == 2)
							orie = 1;
					caledgeang(filterlines[isside], ang[isside], 0, orie);
					lastang = ang[isside];
				}
				else
					lastang = 0;
			}
			else
				lastang = 0;


			//cout << "choose::" << choose << endl;
			//cout << "need_compute_lineangel::" << need_compute_lineangel << endl;


		}
		else {

			/*orie = comparemidside(boundRect,contour, crop,noside, height, int(width * 3 / 14));
			findLines(sideground[0], filterlines[0]);
			caledgeang(filterlines[0], ang[0], 1, orie);
			cout << "mid orie::" << orie << endl;
			if (orie == 0)
				lastang = pi;
			else
				lastang = ang[0];*/
			isside = 0;

			if (boundRect[0].br().y >= height - 5) {
				vector<vector<Point>> allcontours;
				vector<Point> allcontour;
				int index = 0;

				extractbox(labelground, allcontours, index, 1);
				if (allcontours.size() == 0|| boundRect[0].height<height/3) {
					lastang = pi;

				}
				else {
					allcontour = allcontours[index];
					cv::Moments moment;
					cv::Mat temp(allcontour);
					moment = moments(temp, false);
					cv::Point pt;
					if (moment.m00 != 0)
					{

						pt.x = cvRound(moment.m10 / moment.m00);
						pt.y = cvRound(moment.m01 / moment.m00);
						circle(src, pt, 5, Scalar(255, 0, 0), -1);
						float k = (-pt.x + width / 2) / (pt.y - height);

						lastang = atan(k);

					}
					else {
						lastang = pi;
					}
				}
				//	cout << "out side!!!" << endl;
			}
			/*else if (boundRect[0].br().y >= height * 2 / 3)
			{
				lastang = pi;

			}
			else if (boundRect[0].br().y >= height / 2) {*/
			else {

				distance = height - boundRect[0].y - boundRect[0].height;
				if (boundRect[0].br().x <= crop[0].cols / 2) {
					
					orie = 1;
				}
				else if (boundRect[0].tl().x > crop[0].cols / 2)
					orie = 2;
				else {
					
					orie = comparemidside(boundRect, contour, crop, noside, height, int(width * 3 / 14));
				}
				if (boundRect[0].x > fcrop.cols / 3 && boundRect[0].x + boundRect[0].width < fcrop.cols * 2 / 3)
				{
					if (orie == 1)
						lastang = pi / 2;
					else if (orie == 2)
						lastang = pi / -2;

				}
				else {
					findLines(sideground[0], filterlines[0]);
					caledgeang(filterlines[0], ang[0], 1, orie);
					if (orie == 0) {
						lastang = pi;

					}
					else {
						lastang = ang[0];

					}
				}

			}

		}

	}
	else
	{
		lastang = pi;
		isside = -1;

	}

	double area_ratio = cv::sum(fcrop / 255)[0] / (fcrop.cols*fcrop.rows);

	if (area_ratio > 0.95)
		inside = 0;
	else if (area_ratio < 0.1)
		inside = 3;
	else {
		if (isside > 0)
			inside = isside;
		else
			inside = 0;
	}

	//for (int k = 0; k < 3; k++)
	//{
	//	rectangle(src_wrap_img(roi[k]), Point(boundRect[k].x, boundRect[k].y), Point(boundRect[k].x + boundRect[k].width, boundRect[k].y + boundRect[k].height), Scalar(255, 0, 0), 2, 8);
	//	for (size_t i = 0; i < filterlines[k].size(); i++)
	//	{
	//		//cout << filterlines[k].size() << endl;
	//		Vec4i points = filterlines[k][i];

	//		//line(src(roi[k]), Point(points[0], points[1]), Point(points[2],points[3]), Scalar(0,0,255-i*int(255/filterlines[k].size())), 2, CV_AA);	
	//		line(src_wrap_img(roi[k]), Point(points[0], points[1]), Point(points[2], points[3]), Scalar(0,0, 255 - i * int(200 / filterlines[k].size())), 2, CV_AA);
	//	}
	//}

	//cout << "ang::" << lastang * 180 / pi << endl;
	//cv::imshow("t", src_wrap_img);
	////////float tt = lastang * 180 / pi;
	////////string name = "31out/3_" + to_string(num) + "_" + to_string(tt) + ".jpg";
	////////cv::imwrite(name,src_wrap_img );
	//cv::waitKey(0);

}
void follow_edge()
{
	float ang = 0;
	int isside = -1;
	int distance = 0;
	int inside = 0;
	float ratio = 0.2;

	Vision_edge_extraction edge_extraction;
	string zmqip = "tcp://127.0.0.1:10007";
	zmq::context_t ctx{ 1 };

	zmq::socket_t publisher(ctx, zmq::socket_type::pub);
	publisher.bind(zmqip);

	Mat src;
	string f = "41/";

	//加载图片
	for (int i =1; i <=2000; i = i + 1) {

		//string p = f + to_string(i) + ".jpg";
		//string p2 = f + to_string(i-1) + ".jpg";
		
		string p = f + "4_" + to_string(i) + ".png-out.jpg";
		cout << p << endl;
		src = imread(p, 1);

		//检测是否加成功
		if (!src.data)
		{
			//cout << "Could not open or find the image" << endl;
			continue;
		}

		cv::resize(src, src, cv::Size(640, 360));
		CaledgeInfo(src, isside, ang, distance,inside, state.side_advise);

		//cout << "isside:" << isside << endl;
		//cout << "ang1::" << ang * 180 / pi << endl;
		if (isside != -1 && isside == state.lastside) {

			if (isside != 0)
			{
				//cout << " state.lastang" << state.lastang << endl;
				ang = state.lastang*0.8 + ang * 0.2;

			}
			else
			{
				if (ang <= -pi/4)
					state.side_advise = 1;
				else if (ang >= pi / 4)
					state.side_advise = 2;
			}
			state.lastang = ang;
			state.lastside = isside;

		}
		else
		{
			if (isside == -1)
			{

				state.nosidecount++;
				if (state.nosidecount < 5) {
					ang = 0;
					state.lastang = ang;
				}
				if (state.nosidecount < 8 && state.nosidecount >= 5)
				{

					if (state.lastside == 0)
						ang = pi;
					else if (state.lastside == 1)
						ang = pi / 6;
					else if (state.lastside == 2)
						ang = -1 * pi / 6;
				}
				else if (state.nosidecount >= 8) {
					
					state.lastang = 0;
					state.lastside = isside;
				}
			}
			else {
				state.nosidecount = 0;
				state.lastang = ang;
				state.lastside = isside;
			}


		}
		edge_extraction.data.angle = ang;
		edge_extraction.data.isside = inside;
		if (isside != 0)
			edge_extraction.data.distance =0 ;
		else
			edge_extraction.data.distance = convert2realdistance(180- distance);
		zmq::message_t message;
		string topic = "edge_extraction";

		zmq::message_t message_topic(topic.data(), topic.size());

		publisher.send(message_topic, zmq::send_flags::sndmore);

		zmq::message_t message_buffer(sizeof(edge_extraction.buffer));

		memcpy(message_buffer.data(), edge_extraction.buffer, sizeof(edge_extraction.buffer));

		publisher.send(message_buffer, zmq::send_flags::none);
		//cout << "ang2:::" << ang * 180 / pi << endl;
		//Mat src2 = imread(p2, CV_LOAD_IMAGE_COLOR);
		//imshow("src2",src2);
		//cv::waitKey(0);
	}
}

void compute_edgeinfo() {

	
	
	Vision_detect_boundary detect_boundary;
	
	string zmqip = "tcp://127.0.0.1:10006";
	zmq::context_t ctx{ 1 };

	zmq::socket_t publisher(ctx, zmq::socket_type::pub);
	publisher.bind(zmqip);
	Mat src;

	string f ="undistort/";

	//加载图片
	for (int i = 658; i <= 1000; i = i + 2) {

		string p = f + to_string(i) + ".jpg";
		
		src = imread(p, 1);

		//检测是否加成功
		if (!src.data)
		{
			//cout << "Could not open or find the image" << endl;
			return;
		}

		

		CalboundaryInfo(src, detect_boundary);

		zmq::message_t message;
		string topic = "detect_boundary";

		zmq::message_t message_topic(topic.data(), topic.size());

		publisher.send(message_topic, zmq::send_flags::sndmore);

		zmq::message_t message_buffer(sizeof(detect_boundary.buffer));

		memcpy(message_buffer.data(), detect_boundary.buffer, sizeof(detect_boundary.buffer));

		publisher.send(message_buffer, zmq::send_flags::none);
	}
						
	
}
int main(int argc,char *argv[])
{
	zmq::context_t ctx{ 1 };
	zmq::socket_t sub(ctx, zmq::socket_type::sub);
	sub.connect("tcp://127.0.0.1:10008");
	sub.set(zmq::sockopt::subscribe, "vision_mode");
	while (1) {

		Vision_mode Mode;
		zmq::message_t recv;
		sub.recv(recv, zmq::recv_flags::none);
		sub.recv(recv, zmq::recv_flags::none);
		
		std::memcpy(Mode.buffer, recv.data(), recv.size());
		if (Mode.data.mode)
			follow_edge();//follow edge
		else
			compute_edgeinfo();//random cut

	}
	return 0;

}


