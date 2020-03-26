/*
 *  stereo_match.cpp
 *  calibration
 *
 *  Created by Victor  Eruhimov on 1/18/10.
 *  Copyright 2010 Argus Corp. All rights reserved.
 *
 */

#include <opencv2/opencv.hpp>

using namespace cv;

double sad(uchar *img1, const int &widthBytes1, const int &x1, const int &y1, uchar *img2, const int &widthBytes2, const int &x2, const int &y2, const int &width, const int &height, const int &windowSize)
{
	uchar *data1 = img1 + y1 * widthBytes1 + x1;
	uchar *data2 = img2 + y2 * widthBytes2 + x2;

	double sum = 0.0;
	int windowStep = (windowSize - 1) / 2;
	int windowSize2 = windowSize * windowSize;

	if (x1 < windowStep || x1 > width - windowStep - 1 || y1 < windowStep || y1 > height - windowStep - 1)
		return DBL_MAX;

	if (x2 < windowStep || x2 > width - windowStep - 1 || y2 < windowStep || y2 > height - windowStep - 1)
		return DBL_MAX;

	for (int j = -windowStep; j < windowStep + 1; ++j)
		for (int i = -windowStep; i < windowStep + 1; ++i)
		{
			sum += fabs((float)(*(data1 + widthBytes1 * j + i)) - (float)(*(data2 + widthBytes2 * j + i)));
		}

	return sum;
}

double ncc(uchar *img1, const int &widthBytes1, const int &x1, const int &y1, uchar *img2, const int &widthBytes2, const int &x2, const int &y2, const int &width, const int &height, const int &windowSize)
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

void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

void print_help()
{
    printf("Usage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|var] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"
           "[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");
}

int main(int argc, char** argv)
{
    const char* algorithm_opt = "--algorithm=";
    const char* maxdisp_opt = "--max-disparity=";
    const char* blocksize_opt = "--blocksize=";
    const char* nodisplay_opt = "--no-display=";

    if(argc < 3)
    {
        print_help();
        return 0;
    }
    const char* img1_filename = 0;
    const char* img2_filename = 0;
    const char* intrinsic_filename = 0;
    const char* extrinsic_filename = 0;
    const char* disparity_filename = 0;
    const char* point_cloud_filename = 0;
    
    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3 };
    int alg = STEREO_SGBM;
    int SADWindowSize = 0, numberOfDisparities = 0;
    bool no_display = false;
    
    StereoBM bm;
    StereoSGBM sgbm;
	StereoVar var;
    
    for( int i = 1; i < argc; i++ )
    {
        if( argv[i][0] != '-' )
        {
            if( !img1_filename )
                img1_filename = argv[i];
            else
                img2_filename = argv[i];
        }
        else if( strncmp(argv[i], algorithm_opt, strlen(algorithm_opt)) == 0 )
        {
            char* _alg = argv[i] + strlen(algorithm_opt);
            alg = strcmp(_alg, "bm") == 0 ? STEREO_BM :
                  strcmp(_alg, "sgbm") == 0 ? STEREO_SGBM :
                  strcmp(_alg, "hh") == 0 ? STEREO_HH : 
				  strcmp(_alg, "var") == 0 ? STEREO_VAR : -1;
            if( alg < 0 )
            {
                printf("Command-line parameter error: Unknown stereo algorithm\n\n");
                print_help();
                return -1;
            }
        }
        else if( strncmp(argv[i], maxdisp_opt, strlen(maxdisp_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(maxdisp_opt), "%d", &numberOfDisparities ) != 1 ||
                numberOfDisparities < 1 || numberOfDisparities % 16 != 0 )
            {
                printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
                print_help();
                return -1;
            }
        }
        else if( strncmp(argv[i], blocksize_opt, strlen(blocksize_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(blocksize_opt), "%d", &SADWindowSize ) != 1 ||
                SADWindowSize < 1 || SADWindowSize % 2 != 1 )
            {
                printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
                return -1;
            }
        }
        else if( strcmp(argv[i], nodisplay_opt) == 0 )
            no_display = true;
        else if( strcmp(argv[i], "-i" ) == 0 )
            intrinsic_filename = argv[++i];
        else if( strcmp(argv[i], "-e" ) == 0 )
            extrinsic_filename = argv[++i];
        else if( strcmp(argv[i], "-o" ) == 0 )
            disparity_filename = argv[++i];
        else if( strcmp(argv[i], "-p" ) == 0 )
            point_cloud_filename = argv[++i];
        else
        {
            printf("Command-line parameter error: unknown option %s\n", argv[i]);
            return -1;
        }
    }
    
    if( !img1_filename || !img2_filename )
    {
        printf("Command-line parameter error: both left and right images must be specified\n");
        return -1;
    }
    
    if( (intrinsic_filename != 0) ^ (extrinsic_filename != 0) )
    {
        printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
        return -1;
    }
    
    if( extrinsic_filename == 0 && point_cloud_filename )
    {
        printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
        return -1;
    }
    
    int color_mode = alg == STEREO_BM ? 0 : -1;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);
    Size img_size = img1.size();

	//载入极线校正后的部分摄像机参数
	int widthRectified;
	int heightRectified;
	float deltaLeftX;
	float deltaLeftY;
	float deltaRightX;
	FILE *fp = fopen("newparas.bin", "rb");
	fread(&widthRectified, sizeof(int), 1, fp);
	fread(&heightRectified, sizeof(int), 1, fp);
	fread(&deltaLeftX, sizeof(float), 1, fp);
	fread(&deltaLeftY, sizeof(float), 1, fp);
	fread(&deltaRightX, sizeof(float), 1, fp);
	fclose(fp);

	//载入摄像机内外参数
	fp = fopen("cal_res.txt", "r");
	Mat T(3, 1, CV_64FC1);
	double *dataT = (double *)T.data;
	double fc[2];
	double cc[2];
	fscanf(fp, "%*lf%*lf%*lf%lf%lf%lf", dataT, dataT + 1, dataT + 2);
	fscanf(fp, "%*lf%*lf%*lf%*lf%*lf%*lf%*lf%*lf%*lf%lf%lf%lf%lf", fc, fc + 1, cc, cc + 1);
	double tx = norm(T);
	printf("tx = %lf fc = %lf %lf cc = %lf %lf\n", tx, fc[0], fc[1], cc[0], cc[1]);
	fclose(fp);
	double u0 = cc[0] + deltaLeftX;
	double v0 = cc[1] + deltaLeftY;
	double u0r = cc[0] + deltaRightX;
	double b = -tx;

	//载入左摄像机极线校正坐标映射数据
	Mat rectifyLeftX(heightRectified, widthRectified, CV_32FC1);
	Mat rectifyLeftY(heightRectified, widthRectified, CV_32FC1);
	float *dataRectifyLeftX = (float *)rectifyLeftX.data;
	float *dataRectifyLeftY = (float *)rectifyLeftY.data;
	fp = fopen("coordinateleft.bin", "rb");
	assert(fp);
    fread(dataRectifyLeftX, widthRectified * heightRectified, sizeof(float), fp);
	fread(dataRectifyLeftY, widthRectified * heightRectified, sizeof(float), fp);
	fclose(fp);

	//载入右摄像机极线校正坐标映射数据
	Mat rectifyRightX(heightRectified, widthRectified, CV_32FC1);
	Mat rectifyRightY(heightRectified, widthRectified, CV_32FC1);
	float *dataRectifyRightX = (float *)rectifyRightX.data;
	float *dataRectifyRightY = (float *)rectifyRightY.data;
	fp = fopen("coordinateright.bin", "rb");
	assert(fp);
	fread(dataRectifyRightX, widthRectified * heightRectified, sizeof(float), fp);
	fread(dataRectifyRightY, widthRectified * heightRectified, sizeof(float), fp);
	fclose(fp);

	//进行极线校正
	Mat img1Rectified;
	Mat img2Rectified;
	remap(img1, img1Rectified, rectifyLeftX, rectifyLeftY, INTER_CUBIC);
	remap(img2, img2Rectified, rectifyRightX, rectifyRightY, INTER_CUBIC);


    
    Rect roi1, roi2;
    Mat Q;
    
	int minDisparity = -64;
    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : img1.cols / 16;
    
    bm.state->roi1 = roi1;
    bm.state->roi2 = roi2;
    bm.state->preFilterCap = 31;
    bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 15;
    bm.state->minDisparity = minDisparity;
    bm.state->numberOfDisparities = numberOfDisparities;
    bm.state->textureThreshold = 10;
    bm.state->uniquenessRatio = 10;
    bm.state->speckleWindowSize = 100;
    bm.state->speckleRange = 32;
    bm.state->disp12MaxDiff = 1;
    
    sgbm.preFilterCap = 63;
    sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 15;
    
    int cn = img1.channels();
    
    sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = minDisparity;
    sgbm.numberOfDisparities = numberOfDisparities;
    sgbm.uniquenessRatio = 10;
    sgbm.speckleWindowSize = bm.state->speckleWindowSize;
    sgbm.speckleRange = bm.state->speckleRange;
    sgbm.disp12MaxDiff = 1;
    sgbm.fullDP = alg == STEREO_HH;
	
    var.levels = 3;                                 // ignored with USE_AUTO_PARAMS
    var.pyrScale = 0.5;                             // ignored with USE_AUTO_PARAMS
    var.nIt = 25;
    var.minDisp = -numberOfDisparities; //在disp为正时右图同名点在左图同名点左边，disp为负时右图同名点在左图同名点右边
    var.maxDisp = numberOfDisparities;
    var.poly_n = 3;
    var.poly_sigma = 0.0;
    var.fi = 15.0f;
    var.lambda = 0.03f;
    var.penalization = var.PENALIZATION_TICHONOV;   // ignored with USE_AUTO_PARAMS
    var.cycle = var.CYCLE_V;                        // ignored with USE_AUTO_PARAMS
    var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING ;


	printf("%d %d\n", sgbm.SADWindowSize, sgbm.numberOfDisparities);
    
    Mat disp, disp8, disp32;
    //Mat img1p, img2p, dispp;
    //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
    //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
    
    int64 t = getTickCount();
    if( alg == STEREO_BM )
        bm(img1Rectified, img2Rectified, disp);
    else if( alg == STEREO_VAR ) {
        var(img1Rectified, img2Rectified, disp);
    }
    else if( alg == STEREO_SGBM || alg == STEREO_HH )
        sgbm(img1Rectified, img2Rectified, disp);
    t = getTickCount() - t;
    printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

	//uchar *data1 = (uchar *)img1Rectified.data;
	//uchar *data2 = (uchar *)img2Rectified.data;
	//Mat disparity(heightRectified, widthRectified, CV_32FC1);
	//float *dataDisparity = (float *)disparity.data;
	//for (int j = 175; j < 450; ++j)
	//{
	//	int lastMatchIndex = 0;
	//	for (int i = 250; i < 800; ++i)
	//	{
	//		//double maxCoeff = DBL_MIN;
	//		//int matchIndex = -1;
	//		//for (int x = 0; x < widthRectified; ++x)
	//		//{
	//		//	double coeff = ncc(data1, widthRectified, i, j, data2, widthRectified, x, j, widthRectified, heightRectified, 27);
	//		//	if (coeff > maxCoeff)
	//		//	{
	//		//		matchIndex = x;
	//		//		maxCoeff = coeff;
	//		//	}
	//		//}
	//		double minSAD = DBL_MAX;
	//		int matchIndex = -1;
	//		for (int x = 0; x < widthRectified; ++x)
	//		{
	//			double tempSAD = sad(data1, widthRectified, i, j, data2, widthRectified, x, j, widthRectified, heightRectified, 27);
	//			if (tempSAD < minSAD)
	//			{
	//				matchIndex = x;
	//				minSAD = tempSAD;
	//			}
	//		}
	//		if (matchIndex >= 0)
	//		{
	//			dataDisparity[j * widthRectified + i] = matchIndex - i;
	//			lastMatchIndex = matchIndex;
	//		}
	//	}
	//}
	//normalize(disparity, disparity, 0, 1, CV_MINMAX);
	//imshow("disparityNCC", disparity);
	//imwrite("testNCC.png", disparity * 255);

    //disp = dispp.colRange(numberOfDisparities, img1p.cols);
//	normalize(disp, disp, 0, 255, CV_MINMAX);
//	disp -= 16 * (minDisparity - 1)
//	disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities * 16.0));
	disp.convertTo(disp32, CV_32FC1);
	normalize(disp32, disp32, 0, 1, CV_MINMAX);
    if( !no_display )
    {
        namedWindow("left", 1);
        imshow("left", img1Rectified);
		imwrite("leftrectified.bmp", img1Rectified);
        namedWindow("right", 1);
        imshow("right", img2Rectified);
		imwrite("rightrectified.bmp", img2Rectified);
        namedWindow("disparity", 0);
        imshow("disparity", disp32);
        printf("press any key to continue...");
        fflush(stdout);
        waitKey();
        printf("\n");
    }
    
    if(disparity_filename)
        imwrite(disparity_filename, disp32 * 255);

	//三维重构
	fp = fopen("cloud.asc", "w");
	short *dataDisp = (short *)disp.data;
	for (int j = 0; j < heightRectified; ++j)
		for (int i = 0; i < widthRectified; ++i)
		{
			if (dataDisp[j * widthRectified + i] != 16 * (minDisparity - 1))
			{
				double disparity = dataDisp[j * widthRectified + i] / 16.0;
				float x=(b*(i-u0))/(disparity-(u0-u0r));
				float y=(b*fc[0]*(j-v0))/(fc[1]*(disparity-(u0-u0r))); //fc[0]/fc[1]是长宽比，可以提出来
				float z=(b*fc[0])/(disparity-(u0-u0r));
				if (z < 0 && z > -1000)
					fprintf(fp, "%f %f %f\n", x, y, z);
			}
		}
	fclose(fp);

    return 0;
}
