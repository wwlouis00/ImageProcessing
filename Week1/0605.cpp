#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp> 
#include <iostream>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;

int main()
{
	//Mat 1.儲存圖片 2.儲存矩陣
	Mat image = imread("img/test.png",IMREAD_COLOR); //使用的測試圖片
	Mat erodeMat, dilateMat; //宣告erode侵蝕 dilate擴張
	Mat elementKernel = getStructuringElement(MORPH_CROSS,Size(10,10),Point(-1,-1));
	//侵蝕
	erode(image, erodeMat, elementKernel, Point(-1, -1), 1);
	
	//擴張
	dilate(image, dilateMat, elementKernel, Point(-1, -1), 1);
	
	//顯示圖片
	imshow("Eroded", erodeMat);
	imshow("Dilated", dilateMat);
	imshow("Input", image);
	waitKey(0);
	return 0;

	
}