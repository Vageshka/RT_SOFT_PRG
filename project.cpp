#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/aruco.hpp>

using namespace cv;
using namespace std;

static double QR_CODE_WIDTH = 0.055;
static double MAP_WIDTH = 0.21;
static double MAP_HEIGHT = 0.3;
static double QR_CENTER_X = 0.115;
static double QR_CENTER_Y = 0.051;
static int  CHECKERBOARD[2]{3,7};

void get_camera_params(Mat &cameraMatrix, Mat &distCoeffs)
{
    vector<vector<Point3f> > objpoints;
    vector<vector<Point2f> > imgpoints;

    vector<Point3f> objp;
    for(int i{0}; i<CHECKERBOARD[1]; i++)
    {
        for(int j{0}; j<CHECKERBOARD[0]; j++)
        objp.push_back(Point3f(j,i,0));
    }

    vector<String> images;
    string path = "./img/*.jpg";

    glob(path, images);

    Mat frame, gray;
    vector<Point2f> corner_pts;
    bool success;

    for(int i{0}; i<images.size(); i++)
    {
        frame = imread(images[i]);
        cvtColor(frame,gray,COLOR_BGR2GRAY);

        success = findChessboardCorners(gray, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
        
        if(success)
        {
            TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
            cornerSubPix(gray, corner_pts, Size(11,11), Size(-1,-1), criteria);
                        
            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }
    }
    Mat R, T;
    calibrateCamera(objpoints, imgpoints, Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T);
}

void filter(vector<Point2f> &filtered_traj, vector<Point2f> &last_3_unfiltered, Point2f new_point){
    if(last_3_unfiltered.size() == 3){
        Point2f sum = new_point;
        for(int i=0; i<3; i++)
            sum +=last_3_unfiltered[i];
        filtered_traj.push_back(sum/4);
        last_3_unfiltered.erase(last_3_unfiltered.begin());
        last_3_unfiltered.push_back(new_point);
    }
    else{
        filtered_traj.push_back(new_point);
        last_3_unfiltered.push_back(new_point);
    }
}

void drawTrajectory(vector<Point2f>& trajectory, Mat& image) {
    vector<Point2f> im_coords;
    if(trajectory.size() > 3){
        for(int i=0; i < trajectory.size(); ++i) {
            Point2f p = trajectory[i];
            p.x = 400 + p.x*1000;
            p.y = 400 - p.y*1000;
            im_coords.push_back(p); 
        }
        for (auto point = im_coords.begin(); point < im_coords.end() - 1; ++point) {
            line(image, point[0], point[1], Scalar(0,0,255), 3, LINE_AA, 0);
        }
    }
}

void find_corners(vector<Point> contour, vector<Point> &points)
{
    Point x_max, x_min, y_max, y_min;
    x_max = x_min = y_max = y_min = contour[0];
    for(int i=1; i<contour.size(); i++){
        Point cur = contour[i];
        if(x_max.x < cur.x) x_max = cur;
        if(x_min.x > cur.x) x_min = cur;
        if(y_max.y < cur.y) y_max = cur;
        if(y_min.y > cur.y) y_min = cur;
    }
    points.push_back(x_min);
    points.push_back(y_min);
    points.push_back(x_max);
    points.push_back(y_max);
}

void detect_QR_code(Mat frame, Mat &bbox)
{
    Mat gray, lap, blured;

    cvtColor(frame, gray, COLOR_BGR2GRAY);
    Laplacian(gray, lap, CV_32F, 3);
    convertScaleAbs(lap, gray);
    GaussianBlur(gray, blured, Size(5,5), 0);
    threshold(blured, blured, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    morphologyEx(blured, blured, CV_MOP_CLOSE, getStructuringElement(CV_SHAPE_RECT, Size(3,3)), Point(-1,-1), 2);
    erode(blured, blured, Mat(), Point(-1,-1), 3);
    dilate(blured, blured, Mat(), Point(-1,-1), 2);

    vector<vector<Point>>contours;
    findContours(blured, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    if(contours.size()){
        double max_area = contourArea(contours[0]);
        uint num = 0;
        for(uint i=0; i < contours.size(); i++){
            double contour_area = contourArea(contours[i]);
            if(max_area < contour_area){
                max_area = contour_area;
                num = i;
            }
        }

        vector<Point> corners, orthogonal_corn;
        find_corners(contours[num], corners);

        orthogonal_corn.push_back(Point(50,250)); orthogonal_corn.push_back(Point(50, 50)); 
        orthogonal_corn.push_back(Point(250,50)); orthogonal_corn.push_back(Point(250,250)); 
        
        Mat h = findHomography( corners, orthogonal_corn, RANSAC ), tmpIm; 
        warpPerspective(frame, tmpIm, h, tmpIm.size());

        QRCodeDetector qrDecoder = QRCodeDetector();
        Mat tmp_bbox;
        qrDecoder.detect(tmpIm, tmp_bbox);
        if(!tmp_bbox.empty()){
            Mat h_inverse;
            h_inverse = findHomography(orthogonal_corn, corners, RANSAC);
            perspectiveTransform(tmp_bbox, bbox, h_inverse);
        }
        else bbox.release();
    }
    else bbox.release();
}

int main()
{
    Mat trj_image = imread("coordinate_axes.jpg", IMREAD_COLOR);

    Mat cameraMatrix, distCoeffs;
    get_camera_params(cameraMatrix, distCoeffs);

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        return -1;
    }

    namedWindow("Detection", cv::WindowFlags::WINDOW_AUTOSIZE);
    namedWindow("Trajectory", cv::WindowFlags::WINDOW_AUTOSIZE);
    
    vector<Point2f> trajectory;
    vector<Point2f> last_3_unfiltered;

    while (1)
    {
        Mat frame;
        bool bSuccess = cap.read(frame);
        if (!bSuccess)
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }

        Mat bbox;
        detect_QR_code(frame, bbox);

        vector<Point3f> obj_coords;
        obj_coords.push_back(Point3f(0, 0, 0));
        obj_coords.push_back(Point3f(0, QR_CODE_WIDTH, 0));
        obj_coords.push_back(Point3f(QR_CODE_WIDTH, QR_CODE_WIDTH, 0));
        obj_coords.push_back(Point3f(QR_CODE_WIDTH, 0, 0));
        
        Mat rvec,tvec;

        if(!bbox.empty())
        {
            solvePnP(obj_coords, bbox, cameraMatrix, distCoeffs, rvec, tvec);

            cv::aruco::drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.02);

            Mat R, Rtt;
            Rodrigues(rvec,R);
            transpose(R,R);
            Mat1f temp = (-R*tvec);
            Point2f new_coord = Point2f(temp(0), temp(1));
            filter(trajectory, last_3_unfiltered, new_coord);
        }
        imshow("Trajectory", trj_image);
        drawTrajectory(trajectory, trj_image);

        imshow("Detection", frame);
        if (waitKey(30) == 27)
        {
            break;
        }
    }
    return 0;
}
