//compile command: g++ -o goal_detection goal_detection.cpp `pkg-config opencv --cflags --libs`
//команда для компиляции: g++ -o goal_detection goal_detection.cpp `pkg-config opencv --cflags --libs`

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

void detect_lines (Mat img, Mat cdst)
    {
    Mat dst;
    Canny (img, dst, 50, 200, 3);
    cvtColor (dst, cdst, CV_GRAY2BGR);

    vector <Vec4i> lines;

    HoughLinesP (dst, lines, 1, CV_PI/180, 50, 50, 10 );

    for (size_t i = 0; i < lines.size (); i ++ )
        {
        Vec4i l = lines [i];
        line (cdst, Point (l [0], l [1]), Point (l [2], l [3]), Scalar (0,0,255), 3, CV_AA);
        }
    }

Rect detect_goal (Mat img, Mat& res)
    {
    Mat imgThresholded;

    int iLowH = 0;
    int iHighH = 63;

    int iLowS = 0;
    int iHighS = 111;

    int iLowV = 249;
    int iHighV = 255;

    Mat imgHSV;

    medianBlur (img, img, 3);

    cvtColor (img, imgHSV, COLOR_BGR2HSV);

    inRange (imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), res); //Threshold the image

    //morphological opening (remove small objects from the foreground)
    erode  (res, res, getStructuringElement (MORPH_ELLIPSE, Size (5, 5)));
    dilate (res, res, getStructuringElement (MORPH_ELLIPSE, Size (5, 5)));

    //morphological closing (fill small holes in the foreground)
    dilate (res, res, getStructuringElement (MORPH_ELLIPSE, Size (5, 5)));
    erode  (res, res, getStructuringElement (MORPH_ELLIPSE, Size (5, 5)));

    Mat labels;
    Mat stats;
    Mat centroids;
    cv::connectedComponentsWithStats (res, labels, stats, centroids);

    int ind = 0;
    float curr_max = 0;

    for(int i = 0; i < stats.rows; i++)
        {
        int x = stats.at<int>(Point(0, i));
        int y = stats.at<int>(Point(1, i));
        int w = stats.at<int>(Point(2, i));
        int h = stats.at<int>(Point(3, i));

        if (w * h > curr_max && w * h < img.rows * img.cols)
            {
            curr_max = w * h;
            ind = i;
            }
        }

    int x = stats.at <int> (Point (0, ind));
    int y = stats.at <int> (Point (1, ind));
    int w = stats.at <int> (Point (2, ind));
    int h = stats.at <int> (Point (3, ind));

    Rect rect (x, y, w, h);

    return rect;
    }

int main( int argc, char** argv )
    {
    //VideoCapture cap(0); //capture the video from web cam
    VideoCapture cap ("inp.webm"); //capture the video from web cam

    if ( !cap.isOpened() )  // if not success, exit program
        {
        cout << "Cannot open the web cam" << endl;
        return -1;
        }

    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

    int iLowH = 0;
    int iHighH = 179;

    int iLowS = 0;
    int iHighS = 255;

    int iLowV = 0;
    int iHighV = 255;

    //Create trackbars in "Control" window
    cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
    cvCreateTrackbar("HighH", "Control", &iHighH, 179);

    cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS", "Control", &iHighS, 255);

    cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
    cvCreateTrackbar("HighV", "Control", &iHighV, 255);

    //Mat imgOriginal;
    //bool bSuccess = cap.read(imgOriginal); // read a new frame from video

    while (true)
        {
        Mat imgOriginal;
        bool bSuccess = cap.read(imgOriginal); // read a new frame from video

        if (!bSuccess) //if not success, break loop
            {
            cout << "Cannot read a frame from video stream" << endl;
            break;
            }

        /*Mat imgHSV;
        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
        Mat imgThresholded;
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image*/

        //morphological opening (remove small objects from the foreground)
        //erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        //dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

        //morphological closing (fill small holes in the foreground)
        //dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        //erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

        //imshow("Thresholded Image", imgThresholded); //show the thresholded image

        Mat imgThresholded;
        Rect rect = detect_goal (imgOriginal, imgThresholded);

        Mat lines;
        imgOriginal.copyTo (lines);
        detect_lines (imgOriginal, lines);

        Scalar color (255, 0, 0);
        rectangle (imgOriginal, rect, color, 5);

        imshow ("Thresholded Image", imgThresholded); //show the thresholded image
        imshow ("Original", imgOriginal); //show the original image
        imshow ("Lines", lines); //show the original image

        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
                {
            cout << "esc key is pressed by user" << endl;
            break;
            }
        }

    return 0;

    }
