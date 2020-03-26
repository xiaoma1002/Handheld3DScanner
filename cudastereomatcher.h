#ifndef CUDASTEREOMATCHER_H
#define CUDASTEREOMATCHER_H

#include <opencv2/opencv.hpp>
//#include "globaldefine.h"
#include "sapcameraparameters.h"

class CUDAStereoMatcher
{
public:
	CUDAStereoMatcher(float *parameters, const int &w, const int &h, const int &s);
	~CUDAStereoMatcher();
	void match(float *clouds, float *phaseLeft, float *phaseRight, unsigned char *maskLeft, unsigned char *maskRight);
	void match(sapCameraParameters *params, float *clouds, float *phaseLeft, float *phaseRight, unsigned char *maskLeft, unsigned char *maskRight);
	void matchCPU(float *points, const cv::Mat &phaseLeft, const cv::Mat &phaseRight, const cv::Mat &maskLeft, const cv::Mat &maskRight);
	//
	void matchCPU(sapCameraParameters *params, float *points, const cv::Mat &phaseLeft, const cv::Mat &phaseRight, const cv::Mat &maskLeft, const cv::Mat &maskRight, const cv::Mat &speckleLeft, const cv::Mat &speckleRight)
	void matchCPU(sapCameraParameters *params, float *points, const cv::Mat &phaseLeft, const cv::Mat &phaseRight, const cv::Mat &maskLeft, const cv::Mat &maskRight);
	void matchCPU(sapCameraParameters *params, float *points, const std::vector<cv::Mat> &phaseLeft, const std::vector<cv::Mat> &phaseRight, const cv::Mat &maskLeft, const cv::Mat &maskRight, const bool &isLtoR = true);
	void matchCPURtoL(sapCameraParameters *params, float *points, const cv::Mat &phaseLeft, const cv::Mat &phaseRight, const cv::Mat &maskLeft, const cv::Mat &maskRight);

private:
	//SAD
	double sad(uchar *img1, const int &widthBytes1, const int &x1, const int &y1, uchar *img2, const int &widthBytes2, const int &x2, const int &y2, const int &width, const int &height, const int &windowSize);
	double ncc(uchar *img1, const int &widthBytes1, const int &x1, const int &y1, uchar *img2, const int &widthBytes2, const int &x2, const int &y2, const int &width, const int &height, const int &windowSize);
	size_t width;
	size_t height;
	size_t size;
	size_t sizeBytes;
	float *cameraParameters;
	float *phaseLeftDevice;
	float *phaseRightDevice;
	unsigned char *maskLeftDevice;
	unsigned char *maskRightDevice;
	float *cloudDevice;
	size_t pitchCloud;

};

#endif