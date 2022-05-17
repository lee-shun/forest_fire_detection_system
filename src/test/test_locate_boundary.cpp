#include <opencv2/opencv.hpp>
#include <iostream>
#define BIN_DIV 120

using namespace std;
using namespace cv;

int main() {
  Mat srcImg = imread("haha.jpg");
  Mat xianshi = srcImg.clone();
  Mat redChannel;
  namedWindow("【原图】", WINDOW_NORMAL);
  imshow("【原图】", srcImg);
  Mat grayImg;
  vector<Mat> channels;
  split(srcImg, channels);
  // cvtColor(srcImg,grayImg,COLOR_BGR2GRAY);
  grayImg = channels.at(0);
  redChannel = channels.at(2);
  namedWindow("【灰度图】", WINDOW_NORMAL);
  imshow("【灰度图】", grayImg);
  // 均值滤波
  blur(grayImg, grayImg, Size(20, 20), Point(-1, -1));
  namedWindow("【均值滤波后】", WINDOW_NORMAL);
  imshow("【均值滤波后】", grayImg);
  // 转化为二值图
  Mat midImg1 = grayImg.clone();
  int rowNumber = midImg1.rows;
  int colNumber = midImg1.cols;

  for (int i = 0; i < rowNumber; i++) {
    uchar* data = midImg1.ptr<uchar>(i);  // 取第i行的首地址
    uchar* redData = redChannel.ptr<uchar>(i);
    for (int j = 0; j < colNumber; j++) {
      if (data[j] > BIN_DIV && redData[j] < BIN_DIV / 2)
        data[j] = 255;
      else
        data[j] = 0;
    }
  }
  namedWindow("【二值图】", WINDOW_NORMAL);
  imshow("【二值图】", midImg1);
  Mat midImg2 = midImg1.clone();
  Mat element = getStructuringElement(MORPH_RECT, Size(20, 20));
  morphologyEx(midImg1, midImg2, MORPH_CLOSE, element);
  namedWindow("【开运算后】", WINDOW_NORMAL);
  imshow("【开运算后】", midImg2);
  cout << "midImg1.channel=" << midImg1.channels() << endl;
  cout << "mdiImg1.depth" << midImg1.depth() << endl;
  // 查找图像轮廓
  Mat midImg3 = Mat::zeros(midImg2.rows, midImg2.cols, CV_8UC3);
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(midImg2, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  int index = 0;
  for (; index >= 0; index = hierarchy[index][0]) {
    Scalar color(255, 255, 255);
    drawContours(midImg3, contours, index, color, 0, 8, hierarchy);
  }
  namedWindow("【轮廓图】", WINDOW_NORMAL);
  imshow("【轮廓图】", midImg3);
  Mat midImg4 = midImg3.clone();
  // 创建包围轮廓的矩形边界
  for (int i = 0; i < contours.size(); i++) {
    // 每个轮廓
    vector<Point> points = contours[i];
    // 对给定的2D点集，寻找最小面积的包围矩形
    RotatedRect box = minAreaRect(Mat(points));
    Point2f vertex[4];
    box.points(vertex);
    // 绘制出最小面积的包围矩形
    line(xianshi, vertex[0], vertex[1], Scalar(100, 200, 211), 6, LINE_AA);
    line(xianshi, vertex[1], vertex[2], Scalar(100, 200, 211), 6, LINE_AA);
    line(xianshi, vertex[2], vertex[3], Scalar(100, 200, 211), 6, LINE_AA);
    line(xianshi, vertex[3], vertex[0], Scalar(100, 200, 211), 6, LINE_AA);
    // 绘制中心的光标
    Point s1, l, r, u, d;
    s1.x = (vertex[0].x + vertex[2].x) / 2.0;
    s1.y = (vertex[0].y + vertex[2].y) / 2.0;
    l.x = s1.x - 10;
    l.y = s1.y;

    r.x = s1.x + 10;
    r.y = s1.y;

    u.x = s1.x;
    u.y = s1.y - 10;

    d.x = s1.x;
    d.y = s1.y + 10;
    line(xianshi, l, r, Scalar(100, 200, 211), 2, LINE_AA);
    line(xianshi, u, d, Scalar(100, 200, 211), 2, LINE_AA);
  }

  namedWindow("【绘制的最小面积矩形】", WINDOW_NORMAL);
  imshow("【绘制的最小面积矩形】", xianshi);
  waitKey(0);
  return 0;
}
