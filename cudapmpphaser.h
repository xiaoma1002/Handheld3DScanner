#ifndef __CUDAPMPPHASER_H__
#define __CUDAPMPPHASER_H__

#include "clock.h"
#include "phase.h"
#ifdef ARISCUDA
#include <cuda.h>
#include <cutil_inline.h>
#endif
#include <vector>
#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>

class CUDAPMPPhaser
{
public:
	CUDAPMPPhaser(const int &w, const int &h, const bool &useLocked = true);
	~CUDAPMPPhaser();
	void computePhase(const int &numShift, const std::vector<float> &lambdas, const size_t &srcPitch, unsigned char *src[], const float &threshold);
	void computePhaseCPU(const int &numShift, std::vector<cv::Mat> &src, cv::Mat &phase, cv::Mat &mask, const float &threshold, const int &startIndex = 0);
	void computePhaseCPU(const int &numShift, const std::vector<float> &lambdas, std::vector<cv::Mat> &src, std::vector<cv::Mat> &phase, cv::Mat &mask, const float &threshold, const int &startIndex = 0);
	void computePhaseCPUFiltered(const int &numShift, const std::vector<float> &lambdas, std::vector<cv::Mat> &src, std::vector<cv::Mat> &phase, cv::Mat &mask, const float &threshold, const int &startIndex = 0);
	float *getPhaseDevice(const int &index);
	unsigned char *getMask();
private:
	inline float phaseShift3(const uchar &I1, const uchar &I2, const uchar &I3)
	{
		return atan2f(1.732f * (I3 - I2), 2 * I1 - I2 - I3);
	}
	inline float phaseShiftTriangular4(const uchar &I1, const uchar &I2, const uchar &I3, const uchar &I4)
	{
		float tmp;
		float sig_data;
		if (I1 == I3)
			return 3.1415926f;
		else 
		{
			tmp = (float)(I4-I2)/(I1-I3);
			float sig_data;
			if(tmp>0)
				tmp = tmp/(1+tmp);
			else 
				tmp = tmp/(1-tmp);
		}
		if(I3-I1 == 0)
			sig_data = 0.5f*(1+glm::sign(I4-I2));
		else 
		{
			if(I3-I1>0)
				sig_data = 0;
			else{
				if(I4-I2  == 0)
					sig_data = 1;
				else
					sig_data = glm::sign(I4-I2);
			}
		}
		tmp = tmp-2.0f*sig_data;
		tmp = tmp/2*3.1415926f;
		return -tmp; 

	}
	inline float phaseShift4(const uchar &I1, const uchar &I2, const uchar &I3, const uchar &I4)
	{
		return atan2f(I4 - I2, I1 - I3);
	}
	inline float phaseShift2p1(const uchar &I1, const uchar &I2, const uchar &I3)
	{
		return atan2f(I3 - I1, I3 - I2);//2[-pi,pi] f 不写的话默认为double 慢
	}
	float unwrap(const float &numPeriod, const float &phi, const float &phib)
	{
		float phim = numPeriod * phib;
	//	return phi + 6.2831853f * floor((phim - phi) * 0.159155f);
		return phi + 6.2831853f * floor((phim - phi) * 0.159155f + 0.5f);
	}
	int width;
	int height;
	int size;
	size_t pitch;
	size_t sizePhase;
	unsigned short *gapXIndexDevice[2];
	unsigned short *gapYIndexUpDevice[2];
	unsigned short *gapYIndexDownDevice[2];
	unsigned char *maskDevice;
	float *phaseDevice[2];
	std::vector<sapClock> clocks;
	std::vector<cv::Mat> matPhib;
	std::vector<cv::Mat> matPhibFiltered;
};

#endif