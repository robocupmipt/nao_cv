//compile command: g++ -o goal_detection goal_detection.cpp `pkg-config opencv --cflags --libs`
//команда для компиляции: g++ -o goal_detection goal_detection.cpp `pkg-config opencv --cflags --libs`

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int find_biggest_area_component_ind (Mat img, Mat stats)
    {
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

    return ind;
    }

Rect find_biggest_area_component_rect (Mat img)
    {
    Mat labels;
    Mat stats;
    Mat centroids;
    cv::connectedComponentsWithStats (img, labels, stats, centroids);

    int ind = find_biggest_area_component_ind (img, stats);

    int x = stats.at <int> (Point (0, ind));
    int y = stats.at <int> (Point (1, ind));
    int w = stats.at <int> (Point (2, ind));
    int h = stats.at <int> (Point (3, ind));

    Rect rect (x, y, w, h);

    return rect;
    }

void colored_area_mask (Mat& img, int blur_kernel, int morph_kernel, Mat& res, Scalar low_color, Scalar high_color)
    {
    Mat imgHSV;
    Mat img_copy;
    img.copyTo (img_copy);

    medianBlur (img_copy, img_copy, blur_kernel);

    cvtColor (img_copy, imgHSV, COLOR_BGR2HSV);

    inRange (imgHSV, low_color, high_color, res);

    erode  (res, res, getStructuringElement (MORPH_ELLIPSE, Size (morph_kernel, morph_kernel)));
    dilate (res, res, getStructuringElement (MORPH_ELLIPSE, Size (morph_kernel, morph_kernel)));
    dilate (res, res, getStructuringElement (MORPH_ELLIPSE, Size (morph_kernel, morph_kernel)));
    erode  (res, res, getStructuringElement (MORPH_ELLIPSE, Size (morph_kernel, morph_kernel)));
    }

Rect detect_goal (Mat img, Mat& res)
    {
    int iLowH  = 0;
    int iHighH = 63;
    int iLowS  = 0;
    int iHighS = 111;
    int iLowV  = 249;
    int iHighV = 255;

    Scalar low_color  = Scalar (iLowH,  iLowS,  iLowV);
    Scalar high_color = Scalar (iHighH, iHighS, iHighV);

    colored_area_mask (img, 3, 5, res, low_color, high_color);

    return find_biggest_area_component_rect (res);
    }

void draw_lines (Mat& img, vector <Vec4i> lines, Scalar color = Scalar (0, 0, 255))
    {
    for (size_t i = 0; i < lines.size (); i ++)
        {
        Vec4i l = lines [i];
        line (img, Point (l [0], l [1]), Point (l [2], l [3]), color, 3, CV_AA);
        }
    }

void detect_lines (Mat img, Mat& cdst)
    {
    //black out non-green on heavily blurred image
    Mat field_mask;
    img.copyTo (field_mask);


    //medianBlur (img_th_blurred, img_th_blurred, 11);
    //blur (img_blurred, img_blurred, Size (21, 21));
    //colored_area_mask (img_blurred, )
    // 47 115 H компонента поля
    // 0 97 0 186 62 255 - HSV линий

    colored_area_mask (img, 21, 5, field_mask, Scalar (47, 0, 0), Scalar (145, 255, 255));
    //imshow ("Blurred", img_blurred);
    //bitwise_and (img_copy, img_copy, img_copy, field_mask);

    Mat img_copy;
    img.copyTo (img_copy, field_mask);

    //tune the coefficients of Hough transform

    Mat dst;

    Canny (img_copy, dst, 50, 200, 3);
    cvtColor (dst, cdst, CV_GRAY2BGR);

    imshow ("Blurred", img_copy);

    vector <Vec4i> lines;

    HoughLinesP (dst, lines, 1, CV_PI/180, 50, 50, 10 );

    draw_lines (cdst, lines);
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

    //namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

    /*int iLowH = 0;
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
    cvCreateTrackbar("HighV", "Control", &iHighV, 255);*/

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

        //imshow ("Thresholded Image", imgThresholded); //show the thresholded image
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
