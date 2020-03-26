#include "StdAfx.h"
#include "cudastereomatcher.h"
#ifdef ARISCUDA
#include "stereomatch.h"
#include <cuda.h>
#include <cutil_inline.h>
#endif

CUDAStereoMatcher::CUDAStereoMatcher(float *parameters, const int &w, const int &h, const int &s)
{
	cameraParameters = parameters;
	width = w;
	height = h;
	size = s;
	sizeBytes = size * sizeof(float);

//	cutilSafeCall(cudaMalloc((void **)&cloudDevice, sizeBytes * 3));
}

CUDAStereoMatcher::~CUDAStereoMatcher()
{
//	cutilSafeCall(cudaFree(cloudDevice));
}

void CUDAStereoMatcher::match(float *clouds, float *phaseLeft, float *phaseRight, unsigned char *maskLeft, unsigned char *maskRight)
{

//	cutilSafeCall(cudaHostRegister(dataMaskLeft, size, 0));
//	cutilSafeCall(cudaHostRegister(dataMaskRight, size, 0));

//	cutilSafeCall(cudaHostRegister(dataAbsolutePhaseLeft, sizeBytes, 0));
//	cutilSafeCall(cudaHostRegister(dataAbsolutePhaseRight, sizeBytes, 0));

	printf("paras %f %f %f %f %f %f\n", cameraParameters[0], cameraParameters[1], cameraParameters[2], cameraParameters[3], cameraParameters[4], cameraParameters[5]);

#ifdef ARISCUDA
	CUDA::stereoMatch(cameraParameters, width, height, maskLeft, maskRight, phaseLeft, phaseRight, clouds);
#endif
}

void CUDAStereoMatcher::match(sapCameraParameters *params, float *clouds, float *phaseLeft, float *phaseRight, unsigned char *maskLeft, unsigned char *maskRight)
{
#ifdef ARISCUDA
	CUDA::stereoMatch(params->fc, params->ccLeftRectify, params->ccRightRectify, params->tRectify, width, height, maskLeft, maskRight, phaseLeft, phaseRight, clouds);
#endif
}

void CUDAStereoMatcher::matchCPU(float *points, const cv::Mat &phaseLeft, const cv::Mat &phaseRight, const cv::Mat &maskLeft, const cv::Mat &maskRight)
{
	uchar *dataMaskLeft = (uchar *)maskLeft.data;
	uchar *dataMaskRight = (uchar *)maskRight.data;
	float *dataAbsolutePhaseLeft = (float *)phaseLeft.data;
	float *dataAbsolutePhaseRight = (float *)phaseRight.data;


	#pragma omp parallel for
	for (int j = 0; j < height; ++j) //gcc��������� 260ms
		//	for (int y = 0; y < rowIndicesLeft.size(); ++y)
	{
		//		int j = rowIndicesLeft[y];
		int lastIndex = 0;

		for (int i = 0; i < width; ++i)
		{
			int index = j * width + i;
			points[index * 3] = 0.0f;
			points[index * 3 + 1] = 0.0f;
			points[index * 3 + 2] = 0.0f;
			if (dataMaskLeft[index] == 255)
			{
				float matchLeftX;
				float matchLeftY;
				float matchRightX;
				float matchRightY;
			//	lastIndex = i;
				//lastIndex + 1, �������z������
				//����lastIndex����0 1.2s->2.6s
				//x��Ȼ<i 1.2s->0.35s 2014.4.12
				//δ����У��������ᣬ�п�������[0, i]Ҳ�п�������[i, width]��ƽ�й���ֻ��������[0, i]
				//����У������ƽ�й��ᣬ��Ϊ�����ӳ�һ�£�������ƫ��
				//	for (int x = 0; x < i; ++x)
				for (int x = 0; x < width - 1; ++x)
				{

					int indexRight = j * width + x;

					if (j < height)
					{

						if (dataMaskRight[indexRight] == 255 && dataMaskRight[indexRight + 1] == 255)
						{

							if (dataAbsolutePhaseLeft[index] >= dataAbsolutePhaseRight[indexRight] && 
								dataAbsolutePhaseLeft[index] <= dataAbsolutePhaseRight[indexRight + 1] &&
								fabs(dataAbsolutePhaseLeft[index] - dataAbsolutePhaseRight[indexRight]) < 0.5f)
							{

								float weight1 = dataAbsolutePhaseLeft[index] - dataAbsolutePhaseRight[indexRight];
								float weight2 = dataAbsolutePhaseRight[indexRight + 1] - dataAbsolutePhaseLeft[index];

								matchLeftX = i - cameraParameters[0];
								matchLeftY = j - cameraParameters[1];

								float tempX;
								if (dataAbsolutePhaseRight[indexRight] == dataAbsolutePhaseRight[indexRight + 1])
									tempX = x - cameraParameters[2];
								else
									tempX = (x * weight2 + (x + 1) * weight1) / (weight1 + weight2) - cameraParameters[2];

								matchRightX = tempX;
								matchRightY = j - cameraParameters[3];

								float invDisparity = 1.0f / (matchRightX - matchLeftX);
								points[index * 3] = cameraParameters[4] * (matchRightX + matchLeftX) * invDisparity * 0.5f;
								points[index * 3 + 1] = cameraParameters[4] * matchRightY * invDisparity;
								points[index * 3 + 2] = cameraParameters[4] * cameraParameters[5] * invDisparity;

								lastIndex = x;
								//	dataMatchMapLeft[index] = 255;
								//	dataMatchMapRight[indexRight] = 255;
								//	int disparityIndex = matchRightX.size() - 1;
								//	dataDisparityMap[index] = matchRightX[disparityIndex] - matchLeftX[disparityIndex];

								break;

							}
						}
					}
				}
			}
		}
	}
}










////////////////////////////////////////////////////////
void CUDAStereoMatcher::matchCPU(sapCameraParameters *params, float *points, const cv::Mat &phaseLeft, const cv::Mat &phaseRight, const cv::Mat &maskLeft, const cv::Mat &maskRight, const cv::Mat &speckleLeft, const cv::Mat &speckleRight)
{
	uchar *dataMaskLeft = (uchar *)maskLeft.data;
	uchar *dataMaskRight = (uchar *)maskRight.data;
	uchar *dataSpeckleLeft = (uchar *)speckleLeft.data;
	uchar *dataSpeckleRight = (uchar *)speckleRight.data;
	float *dataAbsolutePhaseLeft = (float *)phaseLeft.data;
	float *dataAbsolutePhaseRight = (float *)phaseRight.data;//c++�﷨ ��ֽṹ��

#pragma omp parallel for
	for (int j = 0; j < height; ++j)//���μ���Լ��
	{
		int lastIndex = 0;
		//maskȥ���ĵ��뽹��ģ���ȵĵ�
		for (int i = 0; i < width; ++i)
		{
			int index = j * width + i;
			points[index * 3] = 0.0f;//��i,j�����x,y,zֵ
			points[index * 3 + 1] = 0.0f;
			points[index * 3 + 2] = 0.0f;

			if (dataMaskLeft[index] == 255)
			{
				float matchLeftX;
				float matchLeftY;
				float matchRightX;
				float matchRightY;
				double SADmin = DBL_MAX;
				double NCCmax = 0;//
				int matchRightint_sad = -1;
				int matchRightint_ncc;//
				for (int x = 0; x < width - 1; ++x)
				{
					int indexRight = j * width + x;

					if (dataMaskRight[indexRight] == 255 && dataMaskRight[indexRight + 1] == 255)
					{

						if (dataAbsolutePhaseLeft[index] >= dataAbsolutePhaseRight[indexRight] &&
							dataAbsolutePhaseLeft[index] <= dataAbsolutePhaseRight[indexRight + 1] &&
							fabs(dataAbsolutePhaseLeft[index] - dataAbsolutePhaseRight[indexRight]) < 0.5f)//�ó�һ�����ܵĶ�Ӧ�㣬����SADֵ
						{//���ü���SAD\NCC�ĺ���
							double SADvalue = sad(dataSpeckleLeft, width, i, j, dataSpeckleRight, width, x, j, width, height, 9);//���ڴ�С 9*9
							double NCCvalue = sad(dataSpeckleLeft, width, i, j, dataSpeckleRight, width, x, j, width, height, 9);//
							if (SADmin > SADvalue)
							{
								SADmin = SADvalue;
								matchRightint_sad = x;
							}
							if (NCCmax < NCCvalue)
							{
								NCCmax = NCCvalue;
								matchRightint_ncc = x;
							}
						}
					}
				}

				if (matchRightint_sad >= 0 && SADmin<5000)//�ų� û�з���Ҫ��ĵ� �� ����SADֵ���ܴ� �����
				{
					int indexRight = j * width + matchRightint_sad;
					float weight1 = dataAbsolutePhaseLeft[index] - dataAbsolutePhaseRight[indexRight];
					float weight2 = dataAbsolutePhaseRight[indexRight + 1] - dataAbsolutePhaseLeft[index];
					//�ҵ�����Ӧ���ҵ����ȷ��λ�ã������ظ���ȷ��
					matchLeftX = i - params->ccLeftRectify[0];
					matchLeftY = j - params->ccLeftRectify[1];
					float tempX;
					if (dataAbsolutePhaseRight[indexRight] == dataAbsolutePhaseRight[indexRight + 1])
						tempX = matchRightint_sad - params->ccRightRectify[0];
					else
						tempX = (matchRightint_sad * weight2 + (x + 1) * weight1) / (weight1 + weight2) - params->ccRightRectify[0];

					matchRightX = tempX;
					matchRightY = j - params->ccRightRectify[1];

					float invDisparity = 1.0f / (matchRightX - matchLeftX);
					points[index * 3] = params->tRectify * (i - params->ccLeftRectify[0]) * invDisparity;
					points[index * 3 + 1] = params->tRectify * params->fc[0] * matchRightY * invDisparity / params->fc[1];
					points[index * 3 + 2] = params->tRectify * params->fc[0] * invDisparity;

					lastIndex = matchRightint_sad;
				}
				
				if ( NCCmax>0.7)//�ų�������������趨��ֵ
				{
					int indexRight = j * width + matchRightint_ncc;
					float weight1 = dataAbsolutePhaseLeft[index] - dataAbsolutePhaseRight[indexRight];
					float weight2 = dataAbsolutePhaseRight[indexRight + 1] - dataAbsolutePhaseLeft[index];
					
					matchLeftX = i - params->ccLeftRectify[0];
					matchLeftY = j - params->ccLeftRectify[1];
					float tempX;
					if (dataAbsolutePhaseRight[indexRight] == dataAbsolutePhaseRight[indexRight + 1])
						tempX = matchRightint_ncc - params->ccRightRectify[0];
					else
						tempX = (matchRightint_ncc * weight2 + (matchRightint_ncc + 1) * weight1) / (weight1 + weight2) - params->ccRightRectify[0];

					matchRightX = tempX;
					matchRightY = j - params->ccRightRectify[1];

					float invDisparity = 1.0f / (matchRightX - matchLeftX);
					points[index * 3] = params->tRectify * (i - params->ccLeftRectify[0]) * invDisparity;
					points[index * 3 + 1] = params->tRectify * params->fc[0] * matchRightY * invDisparity / params->fc[1];
					points[index * 3 + 2] = params->tRectify * params->fc[0] * invDisparity;

					lastIndex = matchRightint_ncc;
				}
			}
		}
	}











}void CUDAStereoMatcher::matchCPU(sapCameraParameters *params, float *points, const cv::Mat &phaseLeft, const cv::Mat &phaseRight, const cv::Mat &maskLeft, const cv::Mat &maskRight)
{
	uchar *dataMaskLeft = (uchar *)maskLeft.data;
	uchar *dataMaskRight = (uchar *)maskRight.data;
	float *dataAbsolutePhaseLeft = (float *)phaseLeft.data;
	float *dataAbsolutePhaseRight = (float *)phaseRight.data;

	#pragma omp parallel for
	for (int j = 0; j < height; ++j) //gcc��������� 260ms
	{
		int lastIndex = 0;

		for (int i = 0; i < width; ++i)
		{
			int index = j * width + i;
			points[index * 3] = 0.0f;
			points[index * 3 + 1] = 0.0f;
			points[index * 3 + 2] = 0.0f;
			if (dataMaskLeft[index] == 255)
			{
				float matchLeftX;
				float matchLeftY;
				float matchRightX;
				float matchRightY;
				for (int x = 0; x < width - 1; ++x)
				{
					int indexRight = j * width + x;

					if (j < height)
					{

						if (dataMaskRight[indexRight] == 255 && dataMaskRight[indexRight + 1] == 255)
						{

							if (dataAbsolutePhaseLeft[index] >= dataAbsolutePhaseRight[indexRight] && 
								dataAbsolutePhaseLeft[index] <= dataAbsolutePhaseRight[indexRight + 1] &&
								fabs(dataAbsolutePhaseLeft[index] - dataAbsolutePhaseRight[indexRight]) < 0.5f)
							{

								float weight1 = dataAbsolutePhaseLeft[index] - dataAbsolutePhaseRight[indexRight];
								float weight2 = dataAbsolutePhaseRight[indexRight + 1] - dataAbsolutePhaseLeft[index];

								matchLeftX = i - params->ccLeftRectify[0];
								matchLeftY = j - params->ccLeftRectify[1];

								float tempX;
								if (dataAbsolutePhaseRight[indexRight] == dataAbsolutePhaseRight[indexRight + 1])
									tempX = x - params->ccRightRectify[0];
								else
									tempX = (x * weight2 + (x + 1) * weight1) / (weight1 + weight2) - params->ccRightRectify[0];

								matchRightX = tempX;
								matchRightY = j - params->ccRightRectify[1];

								float invDisparity = 1.0f / (matchRightX - matchLeftX);
								points[index * 3] = params->tRectify * (i - params->ccLeftRectify[0]) * invDisparity;
								points[index * 3 + 1] = params->tRectify * params->fc[0] * matchRightY * invDisparity / params->fc[1];
								points[index * 3 + 2] = params->tRectify * params->fc[0] * invDisparity;

								lastIndex = x;
								//	dataMatchMapLeft[index] = 255;
								//	dataMatchMapRight[indexRight] = 255;
								//	int disparityIndex = matchRightX.size() - 1;
								//	dataDisparityMap[index] = matchRightX[disparityIndex] - matchLeftX[disparityIndex];
								
								//�����⣬��֪Ϊ��
								//float disparity = matchRightX + params->ccRightRectify[0] - i;
								//points[index * 3] = (params->b * (i - params->u0)) / (disparity - (params->u0 - params->u0r));
								//points[index * 3 + 1] = (params->b * params->fc[0] * (j - params->v0)) / (params->fc[1] * (disparity - (params->u0 - params->u0r))); //fc[0]/fc[1]�ǳ���ȣ����������
								//points[index * 3 + 2] = (params->b * params->fc[0]) / (disparity - (params->u0 - params->u0r));

								break;

							}
						}
					}
				}
			}
		}
	}
}

void CUDAStereoMatcher::matchCPU(sapCameraParameters *params, float *points, const std::vector<cv::Mat> &phaseLeft, const std::vector<cv::Mat> &phaseRight, const cv::Mat &maskLeft, const cv::Mat &maskRight, const bool &isLtoR)
{
	int numFrequencys = phaseLeft.size();
	uchar *dataMaskLeft = (uchar *)maskLeft.data;
	uchar *dataMaskRight = (uchar *)maskRight.data;
	float **dataAbsolutePhaseLeft = new float *[numFrequencys];
	float **dataAbsolutePhaseRight = new float *[numFrequencys];
	for (int i = 0; i < numFrequencys; ++i)
	{
		dataAbsolutePhaseLeft[i] = (float *)phaseLeft[i].data;
		dataAbsolutePhaseRight[i] = (float *)phaseRight[i].data;
	}

	#pragma omp parallel for
	for (int j = 0; j < height; ++j) //gcc��������� 260ms
	{
		float matchLeftX; //������OpenMP��������
		float matchLeftY; //������OpenMP��������
		float sumMatchRightX; //������OpenMP��������
		float *matchRightX = new float[numFrequencys]; //������OpenMP��������
		float matchRightY;
		int lastIndex = 0;

		for (int i = 0; i < width; ++i)
		{
			int index = j * width + i;
			points[index * 3] = 0.0f;
			points[index * 3 + 1] = 0.0f;
			points[index * 3 + 2] = 0.0f;
			if (dataMaskLeft[index] == 255)
			{
				int numFind = 0;
				sumMatchRightX = 0.0f;
				for (int k = 0; k < numFrequencys; ++k)
				{
					for (int x = 0; x < width - 1; ++x)
					{
						int indexRight = j * width + x;

						if (j < height)
						{
							if (dataMaskRight[indexRight] == 255 && dataMaskRight[indexRight + 1] == 255)
							{
								if (dataAbsolutePhaseLeft[k][index] >= dataAbsolutePhaseRight[k][indexRight] && 
									dataAbsolutePhaseLeft[k][index] <= dataAbsolutePhaseRight[k][indexRight + 1] &&
									fabs(dataAbsolutePhaseLeft[k][index] - dataAbsolutePhaseRight[k][indexRight]) < 0.5f)
								{
									++numFind;
									float weight1 = dataAbsolutePhaseLeft[k][index] - dataAbsolutePhaseRight[k][indexRight];
									float weight2 = dataAbsolutePhaseRight[k][indexRight + 1] - dataAbsolutePhaseLeft[k][index];

									matchLeftX = i - params->ccLeftRectify[0];
									matchLeftY = j - params->ccLeftRectify[1];

									float tempX;
									if (dataAbsolutePhaseRight[k][indexRight] == dataAbsolutePhaseRight[k][indexRight + 1])
										tempX = x - params->ccRightRectify[0];
									else
										tempX = (x * weight2 + (x + 1) * weight1) / (weight1 + weight2) - params->ccRightRectify[0];

									matchRightX[k] = tempX;
									matchRightY = j - params->ccRightRectify[1];
									sumMatchRightX += matchRightX[k];
									break;
								}
							}
						}
					}
				}

				float aveMatchRightX = sumMatchRightX / numFrequencys;
				float sumDiff = 0.0;
				for (int k = 0; k < numFrequencys; ++k)
					sumDiff += fabs(aveMatchRightX - matchRightX[k]); //ȥ��ɢ�ҵ�

				if (numFind == numFrequencys && sumDiff < 2.0f)
				{
					float invDisparity = 1.0f / (aveMatchRightX - matchLeftX);
					points[index * 3] = params->tRectify * (i - params->ccLeftRectify[0]) * invDisparity;
					points[index * 3 + 1] = params->tRectify * params->fc[0] * matchRightY * invDisparity / params->fc[1];
					points[index * 3 + 2] = params->tRectify * params->fc[0] * invDisparity;
				}
			}
		}
		delete []matchRightX;
	}
	
}

void CUDAStereoMatcher::matchCPURtoL(sapCameraParameters *params, float *points, const cv::Mat &phaseLeft, const cv::Mat &phaseRight, const cv::Mat &maskLeft, const cv::Mat &maskRight)
{
	uchar *dataMaskLeft = (uchar *)maskLeft.data;
	uchar *dataMaskRight = (uchar *)maskRight.data;
	float *dataAbsolutePhaseLeft = (float *)phaseLeft.data;
	float *dataAbsolutePhaseRight = (float *)phaseRight.data;

	#pragma omp parallel for
	for (int j = 0; j < height; ++j) //gcc��������� 260ms
	{
		int lastIndex = 0;

		for (int i = 0; i < width; ++i)
		{
			int index = j * width + i;
			points[index * 3] = 0.0f;
			points[index * 3 + 1] = 0.0f;
			points[index * 3 + 2] = 0.0f;
			if (dataMaskRight[index] == 255)
			{
				float matchLeftX;
				float matchLeftY;
				float matchRightX;
				float matchRightY;
				for (int x = 0; x < width - 1; ++x)
				{
					int indexLeft = j * width + x;

					if (j < height)
					{

						if (dataMaskLeft[indexLeft] == 255 && dataMaskLeft[indexLeft + 1] == 255)
						{

							if (dataAbsolutePhaseRight[index] >= dataAbsolutePhaseLeft[indexLeft] && 
								dataAbsolutePhaseRight[index] <= dataAbsolutePhaseLeft[indexLeft + 1] &&
								fabs(dataAbsolutePhaseRight[index] - dataAbsolutePhaseLeft[indexLeft]) < 0.5f)
							{

								float weight1 = dataAbsolutePhaseRight[index] - dataAbsolutePhaseLeft[indexLeft];
								float weight2 = dataAbsolutePhaseLeft[indexLeft + 1] - dataAbsolutePhaseRight[index];

								matchRightX = i - params->ccRightRectify[0];
								matchRightY = j - params->ccRightRectify[1];

								float tempX;
								if (dataAbsolutePhaseLeft[indexLeft] == dataAbsolutePhaseLeft[indexLeft + 1])
									tempX = x - params->ccLeftRectify[0];
								else
									tempX = (x * weight2 + (x + 1) * weight1) / (weight1 + weight2) - params->ccLeftRectify[0];

								matchLeftX = tempX;
								matchLeftY = j - params->ccLeftRectify[1];

								float invDisparity = 1.0f / (matchRightX - matchLeftX);
							//	points[index * 3] = params->tRectify * (matchRightX + matchLeftX) * invDisparity * 0.5f;
							//	points[index * 3] = params->tRectify * (x - params->u0) * invDisparity; //�������û�����кã�Ҳû3DScanner�ã������ȣ�����úܽ�����
							//	points[index * 3] = params->tRectify + params->tRectify * (x - params->u0) * invDisparity; //�����������û��һ�кã����Ǻܾ���
								points[index * 3] = params->tRectify * (i - params->ccRightRectify[0]) * invDisparity; //RtoLƥ������ͼ�������ؽ�ʱ���ƽϾ��ȣ�����ͼ�������ؽ�ʱ������(��Ϊ��ƥ��㣬�Ѿ����Ǿ��ȸ�դ)
								points[index * 3 + 1] = params->tRectify * params->fc[0] * matchLeftY * invDisparity / params->fc[1];
								points[index * 3 + 2] = params->tRectify * params->fc[0] * invDisparity;
								////�����⣬��֪Ϊ��
								//float disparity = matchRightX + params->ccRightRectify[0] - x;
								//points[index * 3] = (params->b * (x - params->u0)) / (disparity - (params->u0 - params->u0r));
								//points[index * 3 + 1] = (params->b * params->fc[0] * (j - params->v0)) / (params->fc[1] * (disparity - (params->u0 - params->u0r))); //fc[0]/fc[1]�ǳ���ȣ����������
								//points[index * 3 + 2] = (params->b * params->fc[0]) / (disparity - (params->u0 - params->u0r));
								lastIndex = x;

								break;

							}
						}						
					}
				}
			}
		}
	}
}

double CUDAStereoMatcher::sad(uchar *img1, const int &widthBytes1, const int &x1, const int &y1, uchar *img2, const int &widthBytes2, const int &x2, const int &y2, const int &width, const int &height, const int &windowSize)
{
	uchar *data1 = img1 + y1 * widthBytes1 + x1;
	uchar *data2 = img2 + y2 * widthBytes2 + x2;

	double sum = 0.0;
	int windowStep = (windowSize - 1) / 2;
	int windowSize2 = windowSize * windowSize;
	//ͼ���Ե���ֵĴ���
	if (x1 < windowStep || x1 > width - windowStep - 1 || y1 < windowStep || y1 > height - windowStep - 1)
		return DBL_MAX;

	if (x2 < windowStep || x2 > width - windowStep - 1 || y2 < windowStep || y2 > height - windowStep - 1)
		return DBL_MAX;//
	//SAD,��ľ���ֵ���
	for (int j = -windowStep; j < windowStep + 1; ++j)
	for (int i = -windowStep; i < windowStep + 1; ++i)
	{
		sum += fabs((float)(*(data1 + widthBytes1 * j + i)) - (float)(*(data2 + widthBytes2 * j + i)));
	}

	return sum;//û��return����return 0����
}

double CUDAStereoMatcher::ncc(uchar *img1, const int &widthBytes1, const int &x1, const int &y1, uchar *img2, const int &widthBytes2, const int &x2, const int &y2, const int &width, const int &height, const int &windowSize)
{
	uchar *data1 = img1 + y1 * widthBytes1 + x1;
	uchar *data2 = img2 + y2 * widthBytes2 + x2;

	double sum1 = 0.0;
	double sum2 = 0.0;
	double ave1;
	double ave2;
	int windowStep = (windowSize - 1) / 2;
	int windowSize2 = windowSize * windowSize;
	double product = 0.0;
	double product1 = 0.0;
	double product2 = 0.0;

	if (x1 < windowStep || x1 > width - windowStep - 1 || y1 < windowStep || y1 > height - windowStep - 1)
		return DBL_MIN;

	if (x2 < windowStep || x2 > width - windowStep - 1 || y2 < windowStep || y2 > height - windowStep - 1)
		return DBL_MIN;

	for (int j = -windowStep; j < windowStep + 1; ++j)
	for (int i = -windowStep; i < windowStep + 1; ++i)
	{
		sum1 += *(data1 + widthBytes1 * j + i);
		sum2 += *(data2 + widthBytes2 * j + i);
	}

	ave1 = sum1 / windowSize2;
	ave2 = sum2 / windowSize2;

	for (int j = -windowStep; j < windowStep + 1; ++j)
	for (int i = -windowStep; i < windowStep + 1; ++i)
	{
		product += (*(data1 + widthBytes1 * j + i) - ave1) * (*(data2 + widthBytes2 * j + i) - ave2);
		product1 += (*(data1 + widthBytes1 * j + i) - ave1) * (*(data1 + widthBytes1 * j + i) - ave1);
		product2 += (*(data2 + widthBytes2 * j + i) - ave2) * (*(data2 + widthBytes2 * j + i) - ave2);
	}

	return product / sqrt(product1 * product2);
}
