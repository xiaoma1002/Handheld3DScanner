#include "StdAfx.h"
#include "cudapmpphaser.h"
#ifdef ARISCUDA
#include <cuda_gl_interop.h>
#endif
//#include "predefine.h"
#include "ParasHeader.h"

#define PI 3.14159265f

CUDAPMPPhaser::CUDAPMPPhaser(const int &w, const int &h, const bool &locked)
{
	width = w;
	height = h;
	size = w * h;

#ifdef ARISCUDA
	cutilSafeCall(cudaMalloc((void **)&gapXIndexDevice[0], height * sizeof(unsigned short)));
	cutilSafeCall(cudaMalloc((void **)&gapXIndexDevice[1], height * sizeof(unsigned short)));
	cutilSafeCall(cudaMalloc((void **)&gapYIndexUpDevice[0], width * sizeof(unsigned short)));
	cutilSafeCall(cudaMalloc((void **)&gapYIndexUpDevice[1], width * sizeof(unsigned short)));
	cutilSafeCall(cudaMalloc((void **)&gapYIndexDownDevice[0], width * sizeof(unsigned short)));
	cutilSafeCall(cudaMalloc((void **)&gapYIndexDownDevice[1], width * sizeof(unsigned short)));

	cutilSafeCall(cudaMalloc((void **)&maskDevice, size));
#endif

	sizePhase = size * sizeof(float);
#ifdef ARISCUDA
	if (1)
	{
		cutilSafeCall(cudaMalloc((void **)&phaseDevice[0], sizePhase));
		cutilSafeCall(cudaMalloc((void **)&phaseDevice[1], sizePhase));
	}
	else
	{
		//param width  - Width of matrix transfer (columns in bytes)
		cutilSafeCall(cudaMallocPitch((void **)&phaseDevice[0], &pitch, width * 4, height));
		cutilSafeCall(cudaMallocPitch((void **)&phaseDevice[1], &pitch, width * 4, height));
		printf("pitch = %d\n", pitch);
	}
#endif

	clocks.resize(3);
	if (allowNewAlgorithm)
	{
		matPhib.resize(3);
		for (int i = 0; i < matPhib.size(); ++i)
			matPhib[i].create(h, w, CV_32FC1);
		matPhibFiltered.resize(3);
		for (int i = 0; i < matPhibFiltered.size(); ++i)
			matPhibFiltered[i].create(h, w, CV_32FC1);
	}

}

CUDAPMPPhaser::~CUDAPMPPhaser()
{
#ifdef ARISCUDA
	cutilSafeCall(cudaFree(gapXIndexDevice[0]));
	cutilSafeCall(cudaFree(gapXIndexDevice[1]));
	cutilSafeCall(cudaFree(gapYIndexUpDevice[0]));
	cutilSafeCall(cudaFree(gapYIndexUpDevice[1]));
	cutilSafeCall(cudaFree(gapYIndexDownDevice[0]));
	cutilSafeCall(cudaFree(gapYIndexDownDevice[1]));

	cutilSafeCall(cudaFree(maskDevice));

	cutilSafeCall(cudaFree(phaseDevice[0]));
	cutilSafeCall(cudaFree(phaseDevice[1]));
#endif
}

void CUDAPMPPhaser::computePhase(const int &numShift, const std::vector<float> &lambdas, const size_t &srcPitch, unsigned char *src[], const float &threshold)
{
#ifdef ARISCUDA
	CUDA::phase(numShift, lambdas, width, height, srcPitch, src, gapXIndexDevice, gapYIndexUpDevice, gapYIndexDownDevice, phaseDevice, maskDevice, threshold);
#endif
}

void CUDAPMPPhaser::computePhaseCPU(const int &numShift, const std::vector<float> &lambdas, std::vector<cv::Mat> &src, std::vector<cv::Mat> &phase, cv::Mat &mask, const float &threshold, const int &startIndex)
{
	unsigned char *dataMask = (unsigned char *)mask.data;
	float **dataPhase = new float *[phase.size()];
	for (int i = 0; i < phase.size(); ++i)
		dataPhase[i] = (float *)phase[i].data;
	int numSrcs = startIndex * 2 == src.size() ? src.size() / 2 : src.size();
	unsigned char **dataSrc = new unsigned char *[numSrcs];
	for (int i = 0; i < numSrcs; ++i)
		dataSrc[i] = (unsigned char *)src[startIndex + i].data;
	if (numShift == 3)
	{
		float numPeriod1 = lambdas[1] / fabs(lambdas[0] - lambdas[1]);
		float numPeriod2 = lambdas[0] / fabs(lambdas[1] - lambdas[0]);
		float slope1 = 2.0f * PI / (lambdas[0] / 1024 * width);
		float slope2 = 2.0f * PI / (lambdas[1] / 1024 * width);
		float threshold1 = numPeriod1 * PI * 1.5f;
		float threshold2 = numPeriod2 * PI * 1.5f;
		#pragma omp parallel for
		for (int j = 0; j < height; ++j)
			for (int i = 0; i < width; ++i)
			{
				int index = j * width + i;
				float term1 = 0.866f * (dataSrc[0][index] - dataSrc[1][index]);
				float term2 = dataSrc[2][index] - 0.5f * (dataSrc[0][index] + dataSrc[1][index]);
				float modulation = term1 * term1 + term2 * term2;
				if (modulation < threshold * threshold)
				{
					dataMask[index] = 0;
					dataPhase[0][index] = 0.0f;
					dataPhase[1][index] = 0.0f;
				}
				else
				{
					dataMask[index] = 255;
					float phase1 = phaseShift3(dataSrc[0][index], dataSrc[1][index], dataSrc[2][index]);
					float phase2 = phaseShift3(dataSrc[3][index], dataSrc[4][index], dataSrc[5][index]);
					float phib = phase1 - phase2;
					if (phib < 0.0f)
						phib += 6.2831853f;

					float unwrappedPhase = unwrap(numPeriod1, phase1, phib);
					if (slope1 * i - unwrappedPhase > threshold1)
						dataPhase[0][index] = unwrappedPhase + numPeriod1 * 6.2831853f;
					else
						dataPhase[0][index] = unwrappedPhase;

					unwrappedPhase = unwrap(numPeriod2, phase2, phib);
					if (slope2 * i - unwrappedPhase > threshold2)
						dataPhase[1][index] = unwrappedPhase + numPeriod2 * 6.2831853f;
					else
						dataPhase[1][index] = unwrappedPhase;

				}
			}
	}

	if (numShift == 4)
	{
		float numPeriod1 = lambdas[1] / fabs(lambdas[0] - lambdas[1]);
		float numPeriod2 = lambdas[2] / fabs(lambdas[1] - lambdas[2]);
		float lambda12 = numPeriod1 * lambdas[0];
		float lambda23 = numPeriod2 * lambdas[1];
		float numPeriod3 = lambda12 * lambda23 / fabs(lambda12 - lambda23);
		float numPeriod31 = numPeriod3 / lambdas[0];
		float numPeriod32 = numPeriod3 / lambdas[1];
		float numPeriod33 = numPeriod3 / lambdas[2];
		#pragma omp parallel for
		for (int j = 0; j < height; ++j)
			for (int i = 0; i < width; ++i)
			{
				int index = j * width + i;
				float term1 = dataSrc[2][index] - dataSrc[0][index];
				float term2 = dataSrc[3][index] - dataSrc[1][index];
				float modulation = term1 * term1 + term2 * term2;
				if (modulation < threshold * threshold)
				{
					dataMask[index] = 0;
					dataPhase[0][index] = 0.0f;
					dataPhase[1][index] = 0.0f;
					dataPhase[2][index] = 0.0f;
				}
				else
				{
					dataMask[index] = 255;

					float phase1;
					float phase2;
					float phase3;

					if (!systemParasList[currentSystemIndex].triangularFringe)
					{
						phase1 = phaseShift4(dataSrc[0][index], dataSrc[1][index], dataSrc[2][index], dataSrc[3][index]);
						phase2 = phaseShift4(dataSrc[4][index], dataSrc[5][index], dataSrc[6][index], dataSrc[7][index]);
						phase3 = phaseShift4(dataSrc[8][index], dataSrc[9][index], dataSrc[10][index], dataSrc[11][index]);
					}
					else
					{
						phase1 = phaseShiftTriangular4(dataSrc[0][index], dataSrc[1][index], dataSrc[2][index], dataSrc[3][index]);
						phase2 = phaseShiftTriangular4(dataSrc[4][index], dataSrc[5][index], dataSrc[6][index], dataSrc[7][index]);
						phase3 = phaseShiftTriangular4(dataSrc[8][index], dataSrc[9][index], dataSrc[10][index], dataSrc[11][index]);
					}

					float phib12 = phase1 - phase2;
					if (phib12 < 0.0f)
						phib12 += 6.2831853f;
					float phase12 = unwrap(numPeriod1, phase1, phib12) / numPeriod1;

					float phib23 = phase2 - phase3;
					if (phib23 < 0.0f)
						phib23 += 6.2831853f;
					float phase23 = unwrap(numPeriod2, phase2, phib23) / numPeriod2;

					float phib123 = phase12 - phase23;
					if (phib123 < 0.0f)
						phib123 += 6.2831853f;

					dataPhase[0][index] = unwrap(numPeriod31, phase1, phib123);
					dataPhase[1][index] = unwrap(numPeriod32, phase2, phib123);
					dataPhase[2][index] = unwrap(numPeriod33, phase3, phib123);
				}
			}
	}

//	for (int i = 0; i < phase.size(); ++i)
//		cv::medianBlur(phase[i], phase[i], 5);
}
//ÈýÆµÍâ²å
void CUDAPMPPhaser::computePhaseCPUFiltered(const int &numShift, const std::vector<float> &lambdas, std::vector<cv::Mat> &src, std::vector<cv::Mat> &phase, cv::Mat &mask, const float &threshold, const int &startIndex)
{
	if (allowNewAlgorithm)
	{
		unsigned char *dataMask = (unsigned char *)mask.data;
		float **dataPhase = new float *[phase.size()];
		for (int i = 0; i < phase.size(); ++i)
			dataPhase[i] = (float *)phase[i].data;
		int numSrcs = startIndex * 2 == src.size() ? src.size() / 2 : src.size();
		unsigned char **dataSrc = new unsigned char *[numSrcs];
		for (int i = 0; i < numSrcs; ++i)
			dataSrc[i] = (unsigned char *)src[startIndex + i].data;
		float *phib = (float *)matPhib[0].data;
		if (numShift == 3)
		{
			float numPeriod1 = lambdas[1] / fabs(lambdas[0] - lambdas[1]);
			float numPeriod2 = lambdas[0] / fabs(lambdas[1] - lambdas[0]);
			float slope1 = 2.0f * PI / (lambdas[0] / 1024 * width);
			float slope2 = 2.0f * PI / (lambdas[1] / 1024 * width);
			float threshold1 = numPeriod1 * PI * 1.5f;
			float threshold2 = numPeriod2 * PI * 1.5f;
			#pragma omp parallel for
			for (int j = 0; j < height; ++j)
			for (int i = 0; i < width; ++i)
			{
				int index = j * width + i;
				float term1 = 0.866f * (dataSrc[0][index] - dataSrc[1][index]);
				float term2 = dataSrc[2][index] - 0.5f * (dataSrc[0][index] + dataSrc[1][index]);
				float modulation = term1 * term1 + term2 * term2;
				if (modulation < threshold * threshold)
				{
					dataMask[index] = 0;
					dataPhase[0][index] = 0.0f;
					dataPhase[1][index] = 0.0f;
				}
				else
				{
					dataMask[index] = 255;
					float phase1 = phaseShift3(dataSrc[0][index], dataSrc[1][index], dataSrc[2][index]);
					float phase2 = phaseShift3(dataSrc[3][index], dataSrc[4][index], dataSrc[5][index]);
					phib[index] = phase1 - phase2;
					if (phib[index] < 0.0f)
						phib[index] += 6.2831853f;

					//float unwrappedPhase = unwrap(numPeriod1, phase1, phib[index]);
					//if (slope1 * i - unwrappedPhase > threshold1)
					//	dataPhase[0][index] = unwrappedPhase + numPeriod1 * 6.2831853f;
					//else
					//	dataPhase[0][index] = unwrappedPhase;

					//unwrappedPhase = unwrap(numPeriod2, phase2, phib[index]);
					//if (slope2 * i - unwrappedPhase > threshold2)
					//	dataPhase[1][index] = unwrappedPhase + numPeriod2 * 6.2831853f;
					//else
					//	dataPhase[1][index] = unwrappedPhase;
					
				}
			}

		//	cv::GaussianBlur(matPhib[0], matPhib[0], cv::Size(15, 15), 0);
			cv::bilateralFilter(matPhib[0], matPhibFiltered[0], -1, 0.5, 5);
			phib = (float *)matPhibFiltered[0].data;
			#pragma omp parallel for
			for (int j = 0; j < height; ++j)
				for (int i = 0; i < width; ++i)
				{
					int index = j * width + i;
					float term1 = 0.866f * (dataSrc[0][index] - dataSrc[1][index]);
					float term2 = dataSrc[2][index] - 0.5f * (dataSrc[0][index] + dataSrc[1][index]);
					float modulation = term1 * term1 + term2 * term2;
					if (modulation >= threshold * threshold)
					{
						float phase1 = phaseShift3(dataSrc[0][index], dataSrc[1][index], dataSrc[2][index]);
						float phase2 = phaseShift3(dataSrc[3][index], dataSrc[4][index], dataSrc[5][index]);

						float unwrappedPhase = unwrap(numPeriod1, phase1, phib[index]);
						if (slope1 * i - unwrappedPhase > threshold1)
							dataPhase[0][index] = unwrappedPhase + numPeriod1 * 6.2831853f;
						else
							dataPhase[0][index] = unwrappedPhase;

						unwrappedPhase = unwrap(numPeriod2, phase2, phib[index]);
						if (slope2 * i - unwrappedPhase > threshold2)
							dataPhase[1][index] = unwrappedPhase + numPeriod2 * 6.2831853f;
						else
							dataPhase[1][index] = unwrappedPhase;

					}
				}
		}

		float *phib12 = (float *)matPhib[0].data;
		float *phib23 = (float *)matPhib[1].data;
		float *phib123 = (float *)matPhib[2].data;
		if (numShift == 4)
		{
			float numPeriod1 = lambdas[1] / fabs(lambdas[0] - lambdas[1]);
			float numPeriod2 = lambdas[2] / fabs(lambdas[1] - lambdas[2]);
			float lambda12 = numPeriod1 * lambdas[0];
			float lambda23 = numPeriod2 * lambdas[1];
			float numPeriod3 = lambda12 * lambda23 / fabs(lambda12 - lambda23);
			float numPeriod31 = numPeriod3 / lambdas[0];
			float numPeriod32 = numPeriod3 / lambdas[1];
			float numPeriod33 = numPeriod3 / lambdas[2];
			#pragma omp parallel for
			for (int j = 0; j < height; ++j)
			for (int i = 0; i < width; ++i)
			{
				int index = j * width + i;
				float term1 = dataSrc[2][index] - dataSrc[0][index];
				float term2 = dataSrc[3][index] - dataSrc[1][index];
				float modulation = term1 * term1 + term2 * term2;
				if (modulation < threshold * threshold)
				{
					dataMask[index] = 0;
					dataPhase[0][index] = 0.0f;
					dataPhase[1][index] = 0.0f;
					dataPhase[2][index] = 0.0f;
				}
				else
				{
					dataMask[index] = 255;

					float phase1 = phaseShift4(dataSrc[0][index], dataSrc[1][index], dataSrc[2][index], dataSrc[3][index]);
					float phase2 = phaseShift4(dataSrc[4][index], dataSrc[5][index], dataSrc[6][index], dataSrc[7][index]);
					float phase3 = phaseShift4(dataSrc[8][index], dataSrc[9][index], dataSrc[10][index], dataSrc[11][index]);

					//float phase1 = phaseShiftTriangular4(dataSrc[0][index], dataSrc[1][index], dataSrc[2][index], dataSrc[3][index]);
					//float phase2 = phaseShiftTriangular4(dataSrc[4][index], dataSrc[5][index], dataSrc[6][index], dataSrc[7][index]);
					//float phase3 = phaseShiftTriangular4(dataSrc[8][index], dataSrc[9][index], dataSrc[10][index], dataSrc[11][index]);

					phib12[index] = phase1 - phase2;
					if (phib12[index] < 0.0f)
						phib12[index] += 6.2831853f;
				//	float phase12 = unwrap(numPeriod1, phase1, phib12[index]) / numPeriod1;

					phib23[index] = phase2 - phase3;
					if (phib23[index] < 0.0f)
						phib23[index] += 6.2831853f;
				//	float phase23 = unwrap(numPeriod2, phase2, phib23[index]) / numPeriod2;

					//phib123[index] = phase12 - phase23;
					//if (phib123[index] < 0.0f)
					//	phib123[index] += 6.2831853f;

					//dataPhase[0][index] = unwrap(numPeriod31, phase1, phib123[index]);
					//dataPhase[1][index] = unwrap(numPeriod32, phase2, phib123[index]);
					//dataPhase[2][index] = unwrap(numPeriod33, phase3, phib123[index]);
				}
			}

			cv::bilateralFilter(matPhib[0], matPhibFiltered[0], -1, 0.5, 5);
			cv::bilateralFilter(matPhib[1], matPhibFiltered[1], -1, 0.5, 5);
			phib12 = (float *)matPhibFiltered[0].data;
			phib23 = (float *)matPhibFiltered[1].data;
			#pragma omp parallel for
			for (int j = 0; j < height; ++j)
			for (int i = 0; i < width; ++i)
			{
				int index = j * width + i;
				float term1 = dataSrc[2][index] - dataSrc[0][index];
				float term2 = dataSrc[3][index] - dataSrc[1][index];
				float modulation = term1 * term1 + term2 * term2;
				if (modulation < threshold * threshold)
				{
					dataMask[index] = 0;
					dataPhase[0][index] = 0.0f;
					dataPhase[1][index] = 0.0f;
					dataPhase[2][index] = 0.0f;
				}
				else
				{
					dataMask[index] = 255;

					float phase1 = phaseShift4(dataSrc[0][index], dataSrc[1][index], dataSrc[2][index], dataSrc[3][index]);
					float phase2 = phaseShift4(dataSrc[4][index], dataSrc[5][index], dataSrc[6][index], dataSrc[7][index]);
					float phase3 = phaseShift4(dataSrc[8][index], dataSrc[9][index], dataSrc[10][index], dataSrc[11][index]);

					//float phase1 = phaseShiftTriangular4(dataSrc[0][index], dataSrc[1][index], dataSrc[2][index], dataSrc[3][index]);
					//float phase2 = phaseShiftTriangular4(dataSrc[4][index], dataSrc[5][index], dataSrc[6][index], dataSrc[7][index]);
					//float phase3 = phaseShiftTriangular4(dataSrc[8][index], dataSrc[9][index], dataSrc[10][index], dataSrc[11][index]);

				//	phib12[index] = phase1 - phase2;
				//	if (phib12[index] < 0.0f)
				//		phib12[index] += 6.2831853f;
					float phase12 = unwrap(numPeriod1, phase1, phib12[index]) / numPeriod1;

				//	phib23[index] = phase2 - phase3;
				//	if (phib23[index] < 0.0f)
				//		phib23[index] += 6.2831853f;
					float phase23 = unwrap(numPeriod2, phase2, phib23[index]) / numPeriod2;

					phib123[index] = phase12 - phase23;
					if (phib123[index] < 0.0f)
						phib123[index] += 6.2831853f;

				//	dataPhase[0][index] = unwrap(numPeriod31, phase1, phib123[index]);
				//	dataPhase[1][index] = unwrap(numPeriod32, phase2, phib123[index]);
				//	dataPhase[2][index] = unwrap(numPeriod33, phase3, phib123[index]);
				}
			}

		//	cv::bilateralFilter(matPhib[2], matPhibFiltered[2], -1, 1, 5);
		//	phib123 = (float *)matPhibFiltered[2].data;
			#pragma omp parallel for
			for (int j = 0; j < height; ++j)
			for (int i = 0; i < width; ++i)
			{
				int index = j * width + i;
				float term1 = dataSrc[2][index] - dataSrc[0][index];
				float term2 = dataSrc[3][index] - dataSrc[1][index];
				float modulation = term1 * term1 + term2 * term2;
				if (modulation < threshold * threshold)
				{
					dataMask[index] = 0;
					dataPhase[0][index] = 0.0f;
					dataPhase[1][index] = 0.0f;
					dataPhase[2][index] = 0.0f;
				}
				else
				{
					dataMask[index] = 255;

					float phase1 = phaseShift4(dataSrc[0][index], dataSrc[1][index], dataSrc[2][index], dataSrc[3][index]);
					float phase2 = phaseShift4(dataSrc[4][index], dataSrc[5][index], dataSrc[6][index], dataSrc[7][index]);
					float phase3 = phaseShift4(dataSrc[8][index], dataSrc[9][index], dataSrc[10][index], dataSrc[11][index]);

					//float phase1 = phaseShiftTriangular4(dataSrc[0][index], dataSrc[1][index], dataSrc[2][index], dataSrc[3][index]);
					//float phase2 = phaseShiftTriangular4(dataSrc[4][index], dataSrc[5][index], dataSrc[6][index], dataSrc[7][index]);
					//float phase3 = phaseShiftTriangular4(dataSrc[8][index], dataSrc[9][index], dataSrc[10][index], dataSrc[11][index]);

					//phib12[index] = phase1 - phase2;
					//if (phib12[index] < 0.0f)
					//	phib12[index] += 6.2831853f;
					//float phase12 = unwrap(numPeriod1, phase1, phib12[index]) / numPeriod1;

					//phib23[index] = phase2 - phase3;
					//if (phib23[index] < 0.0f)
					//	phib23[index] += 6.2831853f;
					//float phase23 = unwrap(numPeriod2, phase2, phib23[index]) / numPeriod2;

					//phib123[index] = phase12 - phase23;
					//if (phib123[index] < 0.0f)
					//	phib123[index] += 6.2831853f;

					dataPhase[0][index] = unwrap(numPeriod31, phase1, phib123[index]);
					dataPhase[1][index] = unwrap(numPeriod32, phase2, phib123[index]);
					dataPhase[2][index] = unwrap(numPeriod33, phase3, phib123[index]);
				}
			}
		}
	}

//	for (int i = 0; i < phase.size(); ++i)
//		cv::medianBlur(phase[i], phase[i], 5);

//	for (int i = 0; i < phase.size(); ++i)
//		cv::GaussianBlur(phase[i], phase[i], cv::Size(0, 0), 3.5);
}

void CUDAPMPPhaser::computePhaseCPU(const int &numShift, std::vector<cv::Mat> &src, cv::Mat &phase, cv::Mat &mask, const float &threshold, const int &startIndex)
{
	unsigned char *dataMask = (unsigned char *)mask.data;
	float *dataPhase = (float *)phase.data;
	int numSrcs = startIndex * 2 == src.size() ? src.size() / 2 : src.size();
	unsigned char **dataSrc = new unsigned char *[numSrcs];
	for (int i = 0; i < numSrcs; ++i)
		dataSrc[i] = (unsigned char *)src[startIndex + i].data;

	if (numShift == 4)
	{
		#pragma omp parallel for
		for (int j = 0; j < height; ++j)
			for (int i = 0; i < width; ++i)
			{
				int index = j * width + i;
				float term1 = dataSrc[2][index] - dataSrc[0][index];
				float term2 = dataSrc[3][index] - dataSrc[1][index];
				float modulation = term1 * term1 + term2 * term2;
				if (modulation < threshold * threshold)
				{
					dataMask[index] = 0;
					dataPhase[index] = 0.0f;
				}
				else
				{
					dataMask[index] = 255;

					float phase = phaseShift4(dataSrc[0][index], dataSrc[1][index], dataSrc[2][index], dataSrc[3][index]);

					dataPhase[index] = phase;
				}
			}
	}

	std::vector<int> rowIndices;
	std::vector<int> colIndices;
	for (int j = 0; j < height; ++j)
	{
		for (int i = 0; i < width; ++i)
		{
			int index = j * width + i;
			if (dataSrc[4][index] > 50)
			{
				rowIndices.push_back(j);
				colIndices.push_back(i);
				break;
			}
		}
	}
	for (int j = 0; j < rowIndices.size(); ++j)
	{
		int count = 0;
		int lastIndex = colIndices[j];
		int lastChangeIndex = 0;
		for (int i = colIndices[j]; i < width; ++i)
		{
			int index = rowIndices[j] * width + i;
			int indexRow = rowIndices[j] * width;
			if (dataMask[index] == 255)
			{
				if (dataPhase[indexRow + lastIndex] - dataPhase[index] > 1.5 * PI)// && i - lastChangeIndex > 10)
				{
					++count;
					lastChangeIndex = i;
				}
				lastIndex = i;
				dataPhase[index] = count * 6.2831853f + dataPhase[index];
			}
		}
		
		count = 0;
		lastIndex = colIndices[j];
		lastChangeIndex = width;
		for (int i = colIndices[j]; i >= 0; --i)
		{
			int index = rowIndices[j] * width + i;
			int indexRow = rowIndices[j] * width;
			if (dataMask[index] == 255)
			{
				if (dataPhase[index] - dataPhase[indexRow + lastIndex] > 1.5 * PI)// && lastChangeIndex - i > 10)
				{
					--count;
					lastChangeIndex = i;
				}
				lastIndex = i;
				dataPhase[index] = count * 6.2831853f + dataPhase[index];
			}
		}

	}
}



///////////////////////////////////////////////////////
void CUDAPMPPhaser::computePhaseCPU(std::vector<cv::Mat> &src, cv::Mat &phase, cv::Mat &mask, const float &threshold, const int &startIndex)
{
	unsigned char *dataMask = (unsigned char *)mask.data;
	float *dataPhase = (float *)phase.data;
	int numSrcs = startIndex * 2 == src.size() ? src.size() / 2 : src.size();
	unsigned char **dataSrc = new unsigned char *[numSrcs];
	for (int i = 0; i < numSrcs; ++i)
		dataSrc[i] = (unsigned char *)src[startIndex + i].data;


#pragma omp parallel for
	for (int j = 0; j < height; ++j)
	for (int i = 0; i < width; ++i)
	{
		int index = j * width + i;
		float term1 = dataSrc[2][index] - dataSrc[0][index];
		float term2 = dataSrc[2][index] - dataSrc[1][index];
		float modulation = term1 * term1 + term2 * term2;//
		if (modulation < threshold * threshold)
		{
			dataMask[index] = 0;
			dataPhase[index] = 0.0f;
		}
		else
		{
			dataMask[index] = 255;

			float phase = phaseShift2p1(dataSrc[0][index], dataSrc[1][index], dataSrc[2][index]);

			dataPhase[index] = phase;
		}
	}

}






float *CUDAPMPPhaser::getPhaseDevice(const int &index)
{
	return phaseDevice[index];
}

unsigned char *CUDAPMPPhaser::getMask()
{
	return maskDevice;
}

