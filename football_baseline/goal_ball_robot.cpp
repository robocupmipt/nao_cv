//compile command: g++ -o goal_ball_robot goal_ball_robot.cpp `pkg-config opencv --cflags --libs`
//команда для компиляции: g++ -o goal_ball_robot goal_ball_robot.cpp `pkg-config opencv --cflags --libs`

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "math.h"

using namespace cv;
using namespace std;

int WIND_X = 800;
int WIND_Y = 600;

int GOAL_POS = 100;
int GOAL_LEN = 300;

Scalar FIELD_COLOR = Scalar (10,  240, 10);
Scalar GOAL_COLOR  = Scalar (30,   40, 70);
Scalar BALL_COLOR  = Scalar (120, 130, 30);
Scalar ROBOT_COLOR = Scalar (160,  40, 10);
Scalar TRAJ_COLOR  = Scalar (60,   40, 180);

int BALL_RADIUS   = 20;
int CIRCLE_RADIUS = 50;
int ROBOT_SIZE    = 50;

void draw_scenario (Mat& img, int xr, int yr, int xb, int yb, vector <Point2d> &trajectory)
    {
    rectangle (img, Point (0, 0), Point (WIND_X, WIND_Y), FIELD_COLOR, -1);

    circle    (img, Point (xb, yb), CIRCLE_RADIUS, GOAL_COLOR);
    circle    (img, Point (xb, yb), BALL_RADIUS, BALL_COLOR, -1);

    Point robot_tl = Point (xr - int (ROBOT_SIZE / 2), yr - int (ROBOT_SIZE / 2));
    Point robot_br = Point (xr + int (ROBOT_SIZE / 2), yr + int (ROBOT_SIZE / 2));
    rectangle (img, robot_tl, robot_br, GOAL_COLOR, -1);

    line (img, Point (WIND_X, GOAL_POS), Point (WIND_X, GOAL_POS + GOAL_LEN), GOAL_COLOR, 5);

    for (int i = 0; i < trajectory.size () - 1; i ++)
        line (img, trajectory [i], trajectory [i + 1], TRAJ_COLOR, 3);
    }

void find_trajectory_test (int xr, int yr, int xb, int yb, vector <Point2d> &trajectory)
    {
    trajectory.clear ();
    trajectory.push_back (Point2d (xr, yr));
    trajectory.push_back (Point2d (xb, yb));
    }

void find_trajectory (int xr, int yr, int xb, int yb, int max_step, vector <Point2d> &trajectory)
    {
    trajectory.clear ();

    trajectory.push_back (Point2d (xr, yr));

    //-----------------------------------------------------------
    //find starting point on the circle

    int xbr = xb - xr; //x ball relative
    int ybr = yb - yr; //y ball relative

    int r    = CIRCLE_RADIUS;
    float leng = sqrt (xbr*xbr + ybr*ybr);

    float beta  = asin (float (ybr) / leng);
    float alpha = asin (float (r) / leng);

    if (xb < xr)
        {
        float sx = 0;
        float sy = 0;

        if (yr + (yr - GOAL_POS - int (GOAL_LEN / 2)) * (xr - xb) / (WIND_X - xb) > yb)
            {
            sx = - leng * cos (alpha + beta) * cos (alpha) + xr;
            sy = leng * sin (alpha + beta) * cos (- alpha) + yr;
            }

        else
            {
            alpha = - alpha;

            sx = - leng * cos (alpha + beta) * cos (alpha) + xr;
            sy = leng * sin (alpha + beta) * cos (- alpha) + yr;
            }

        trajectory.push_back (Point2d (int (sx), int (sy)));
        }

    //-----------------------------------------------------------
    //find kick point on the circle
    int gbx = xb - WIND_X;                        //goal-ball x
    int gby = yb - GOAL_POS - int (GOAL_LEN / 2); //goal-ball y

    int length_gb = int (sqrt (gbx*gbx + gby*gby));

    int kpx = xb + int (CIRCLE_RADIUS * gbx / length_gb); //kick point x
    int kpy = yb + int (CIRCLE_RADIUS * gby / length_gb); //kick point y

    trajectory.push_back (Point2d (kpx, kpy));
    trajectory.push_back (Point2d (WIND_X, GOAL_POS + int (GOAL_LEN / 2)));
    }

int main( int argc, char** argv )
    {
    namedWindow ("Simulation", CV_WINDOW_AUTOSIZE); //create a window called "Control"

    Mat img (WIND_Y, WIND_X, CV_8UC3, Scalar (0, 0, 0));

    int xr = int (1 * WIND_X / 5);
    int yr = int (WIND_Y / 3);

    int xb = int (WIND_X / 2);
    int yb = int (WIND_Y / 2);

    bool upd = true;

    vector <Point2d> trajectory;

    while (true)
        {
        if (upd == true)
            {
            //find_trajectory_test (xr, yr, xb, yb, trajectory);
            find_trajectory (xr, yr, xb, yb, 20, trajectory);

            upd = false;
            }

        draw_scenario (img, xr, yr, xb, yb, trajectory);

        imshow ("Simulation", img);

        char keyb = waitKey (100);

        if (keyb != -1)
            {
            upd = true;
            }

        if (keyb == 'q') break;

        else if (keyb == 't') yr -= 5;
        else if (keyb == 'f') xr -= 5;
        else if (keyb == 'g') yr += 5;
        else if (keyb == 'h') xr += 5;
        else if (keyb == 'i') yb -= 5;
        else if (keyb == 'j') xb -= 5;
        else if (keyb == 'k') yb += 5;
        else if (keyb == 'l') xb += 5;

        if (keyb == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
            {
            cout << "esc key is pressed by user" << endl;
            break;
            }
        }

    return 0;
    }
