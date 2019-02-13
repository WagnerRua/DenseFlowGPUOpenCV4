#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudacodec.hpp"


#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace cv::cuda;
using namespace std;

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
       double lowerBound, double higherBound) {
	#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
	for (int i = 0; i < flow_x.rows; ++i) {
		for (int j = 0; j < flow_y.cols; ++j) {
			float x = flow_x.at<float>(i,j);
			float y = flow_y.at<float>(i,j);
			img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
			img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
		}
	}
	#undef CAST
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color){
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

int main(int argc, char** argv){
	// IO operation
	const String keys =
		"{ f  | vidFile    | ex.avi  | filename of video }"
		"{ x  | xFlowFile  | x       | filename prefix of flow x component }"
		"{ y  | yFlowFile  | y       | filename prefix of flow x component }"
		"{ i  | imgFile    | i       | filename prefix of image }"
		"{ b  | bound      | 20      | specify the maximum (px) of optical flow }"
		"{ t  | type       | 1       | specify the optical flow algorithm }"
		"{ d  | device_id  | 0       | specify gpu id }"
		"{ s  | step       | 1       | specify the step for frame sampling }"
		"{ h  | height     | 0       | specify the height of saved flows, 0: keep original height }"
		"{ w  | width      | 0       | specify the width of saved flows,  0: keep original width }"    
		;

	CommandLineParser cmd(argc, argv, keys);
	String vidFile = cmd.get<String>("f");
	String xFlowFile = cmd.get<String>("x");
	String yFlowFile = cmd.get<String>("y");
	String imgFile = cmd.get<String>("i");
	int bound = cmd.get<int>("b");
        int type  = cmd.get<int>("t");
        int device_id = cmd.get<int>("d");
        int step = cmd.get<int>("s");
        int height = cmd.get<int>("h");
        int width = cmd.get<int>("w");


	VideoCapture capture(vidFile);
	if(!capture.isOpened()) {
		printf("Could not initialize capturing..\n");
		return -1;
	}

	int frame_num = 0;
	int new_width;
	Mat image, prev_image, prev_grey, grey, frame, flow_x, flow_y;
	GpuMat frame_0, frame_1;
	GpuMat d_flow;

	setDevice(device_id);

    	Ptr<cuda::BroxOpticalFlow> alg_brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
    	Ptr<cuda::FarnebackOpticalFlow> alg_farn = cuda::FarnebackOpticalFlow::create();
    	Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1 = cuda::OpticalFlowDual_TVL1::create();


	while(true) {
		capture >> frame;
		if(frame.empty())
			break;
		new_width = (frame.cols * 256) / frame.rows ;
		resize(frame, frame, Size(new_width , 256), 0, 0, INTER_CUBIC);
		if(frame_num == 0) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_image.create(frame.size(), CV_8UC3);
			prev_grey.create(frame.size(), CV_8UC1);

			frame.copyTo(prev_image);
			cvtColor(prev_image, prev_grey, COLOR_BGR2GRAY);

			frame_num++;

			int step_t = step;
			while (step_t > 1){
				capture >> frame;
				step_t--;
			}
			continue;
		}

		frame.copyTo(image);
		cvtColor(image, grey, COLOR_BGR2GRAY);


		frame_0.upload(prev_grey);
		frame_1.upload(grey);


                // GPU optical flow
		switch(type){
		case 0:
			alg_farn->calc(frame_0, frame_1, d_flow);
			break;
		case 1:
			alg_tvl1->calc(frame_0, frame_1, d_flow);
			break;
		case 2:
			GpuMat d_frame0f, d_frame1f;
	                frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
	                frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
			alg_brox->calc(d_frame0f, d_frame1f, d_flow);
			break;
		}

 		GpuMat planes[2];
        	cuda::split(d_flow, planes);

		Mat flow_x(planes[0]);
        	Mat flow_y(planes[1]);

		// Output optical flow
		Mat imgX(flow_x.size(),CV_8UC1);
		Mat imgY(flow_y.size(),CV_8UC1);
		convertFlowToImage(flow_x,flow_y, imgX, imgY, -bound, bound);
		char tmp[20];
		sprintf(tmp,"_%06d.jpg",int(frame_num));

		Mat imgX_, imgY_, image_;
                height = (height > 0)? height : imgX.rows;
                width  = (width  > 0)? width  : imgX.cols;
		resize(imgX,imgX_,cv::Size(width,height));
		resize(imgY,imgY_,cv::Size(width,height));
		resize(image,image_,cv::Size(width,height));

		imwrite(xFlowFile + tmp,imgX_);
		imwrite(yFlowFile + tmp,imgY_);
		imwrite(imgFile + tmp, image_);

		std::swap(prev_grey, grey);
		std::swap(prev_image, image);
		frame_num = frame_num + 1;

		int step_t = step;
		while (step_t > 1){
			capture >> frame;
			step_t--;
		}
	}
	return 0;
}
