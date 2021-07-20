#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

const static double QR_CODE_WIDTH = 0.06;
const static double MAP_WIDTH = 0.21;
const static double MAP_HEIGHT = 0.3;
const static double QR_CENTER_X = 0.115;
const static double QR_CENTER_Y = 0.051;
const static std::string path = "./images_User_Facing/";


void findCorners(const vector<Point>& contours, const Point& points[4]) {
    Point &x_max=points[2], &x_min=points[0], &y_max=points[3], &y_min=points[1];
    x_max = x_min = y_max = y_min = contours[0];

    for (auto contour : contours){
        if (x_max.x < cur.x) x_max = contour;
        if (x_min.x > cur.x) x_min = contour;
        if (y_max.y < cur.y) y_max = contour;
        if (y_min.y > cur.y) y_min = contour;
    }
}

void display(Mat &im, Mat &bbox) {
    for (int i = 0; i < bbox.rows; i++) {
        line(im, Point2i(bbox.at<float>(i, 0), bbox.at<float>(i, 1)), 
        Point2i(bbox.at<float>((i + 1) % n, 0), bbox.at<float>((i + 1) % n, 1)), Scalar(255, 0, 0), 3);
    }

    imshow("Detection", im);
}

void getCameraParams(Mat& cameraMatrix, Mat& distCoeffs) {
    // Defining the dimensions of checkerboard
    int CHECKERBOARD[2]{7, 7}; 

    // Creating vector to store vectors of 3D points for each checkerboard image
    vector<vector<Point3f>> objpoints;

    // Creating vector to store vectors of 2D points for each checkerboard image
    vector<vector<Point2f>> imgpoints;

    // Defining the world coordinates for 3D points
    vector<Point3f> objp;
    for(int i = 0; i < CHECKERBOARD[1]; ++i) {
        for(int j= 0; j < CHECKERBOARD[0]; ++j)
        objp.push_back(Point3f(j, i, 0));
    }

    // Extracting path of individual image stored in a given directory
    vector<cv::String> images;
    glob(path, images);

    Mat frame, gray;
    // vector to store the pixel coordinates of detected checker board corners 
    vector<Point2f> cornerPts;
    bool success;

    // Looping over all the images in the directory
    for (auto image: images) {
        frame = imread(image);
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Finding checker board corners
        // If desired number of corners are found in the image then success = true  
        success = findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), cornerPts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        
        /* 
        * If desired number of corner are detected,
        * we refine the pixel coordinates and display 
        * them on the images of checker board
        */
        if (success) {
            TermCriteria criteria(TermCriteria::EPS | TermCriteria::COUNT, 30, 0.001);
            
            // refining pixel coordinates for given 2d points.
            cornerSubPix(gray, cornerPts, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            
            objpoints.push_back(objp);
            imgpoints.push_back(cornerPts);
        }
    }

    Mat R, T;

    /*
    * Performing camera calibration by 
    * passing the value of known 3D points (objpoints)
    * and corresponding pixel coordinates of the 
    * detected corners (imgpoints)
    */
    calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);
}

void drawTrajectory(const vector<Point2f>& trajectory, Mat& image) {
    for (auto point = trajectory.begin(); point < trajectory.end() - 1; ++point) {
        line(image, point[0], point[1], Scalar(0, 0, 255), 3, LINE_AA, 0);
    }
}

int main() {
    Mat trjImage = imread("map.jpeg", IMREAD_COLOR);
    resize(trjImage, trjImage, Size(trjImage.cols / 2, trjImage.rows / 2));

    vector<Point2f> trajectory;
    Mat cameraMatrix, distCoeffs;

    getCameraParams(cameraMatrix, distCoeffs);
    VideoCapture cap(1);

    if (!cap.isOpened()) {
        return -1;
    }

    namedWindow("Detection", WindowFlags::WINDOW_AUTOSIZE);
    namedWindow("Trajectory", WindowFlags::WINDOW_AUTOSIZE);
    namedWindow("Homography", WindowFlags::WINDOW_AUTOSIZE);

    while (1) {
        Mat frame;
        bool bSuccess = cap.read(frame);
        if (!bSuccess) {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }

        Mat gray, lap, blured;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        Laplacian(gray, lap, CV_32F, 3);
        convertScaleAbs(lap, gray);
        GaussianBlur(gray, blured, Size(5,5), 0);
        threshold(blured, blured, 0, 255, THRESH_BINARY | THRESH_OTSU);
        morphologyEx(blured, blured, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3,3)), Point(-1,-1), 2);
        erode(blured, blured, Mat(), Point(-1,-1), 3);
        dilate(blured, blured, Mat(), Point(-1,-1), 2);
        morphologyEx(blured, blured, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3,3)), Point(-1,-1), 2);
        
        vector<vector<Point>> contours;
        findContours(blured, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (contours.size()){
            Rect max_br = boundingRect(contours[0]);
            uint num = 0;
            for (uint i = 0; i < contours.size(); ++i){
                Rect br = boundingRect(contours[i]);
                if (br.area() > max_br.area()){
                    max_br = br;
                    num = i;
                }
            }

            Point corners[4];
            find_corners(contours[num], corners);
            vector<Point> p, x;
            for(int i = 0; i < 4; i++)
                x.push_back(corners[i]);
            p.push_back(Point(50,250)); p.push_back(Point(50, 50)); 
            p.push_back(Point(250,50)); p.push_back(Point(250, 250)); 
            Mat h = findHomography( x, p, RANSAC), im; 
            warpPerspective(frame, im, h, im.size());
            imshow("Homography", im);
        }

        QRCodeDetector qrDecoder = QRCodeDetector();
        
        Mat bbox, rectifiedImage;
        qrDecoder.detect(frame, bbox);
        // std::string data = qrDecoder.detectAndDecode(frame, bbox, rectifiedImage);

        if (!bbox.empty()) {
            vector<Point3f> objCoords;
            objCoords.push_back(Point3f(0, 0, 0));
            objCoords.push_back(Point3f(0, QR_CODE_WIDTH, 0));
            objCoords.push_back(Point3f(QR_CODE_WIDTH, QR_CODE_WIDTH, 0));
            objCoords.push_back(Point3f(QR_CODE_WIDTH, 0, 0));
            Mat R,T;

            solvePnP(objCoords, bbox, cameraMatrix, distCoeffs, R, T);
            display(im, bbox);
            Mat Rt, Rtt;
            Rodrigues(R, Rt);
            transpose(Rt, Rtt);
            cout << -Rtt*T << endl;
            Mat1f temp = -Rtt*T;
            Point2f point((temp(0) + QR_CENTER_X) * (trjImage.cols / MAP_WIDTH), (temp(1) + QR_CENTER_Y)*(trjImage.rows / MAP_HEIGHT));
            trajectory.push_back(point);
        }

        imshow("Trajectory", trjImage);
        if (trajectory.size()) {
            drawTrajectory(trajectory, trjImage);
        }

        imshow("Detection", frame);
        if (waitKey(30) == 27) {
            break;
        }
    }
    return 0;
}
