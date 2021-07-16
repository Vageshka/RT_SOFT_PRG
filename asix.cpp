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

static double QR_CODE_WIDTH = 0.06;
static double MAP_WIDTH = 0.21;
static double MAP_HEIGHT = 0.3;
static double QR_CENTER_X = 0.115;
static double QR_CENTER_Y = 0.051;

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
    // Path of the folder containing checkerboard images
    std::string path = "./images_User_Facing/";

    glob(path, images);

    Mat frame, gray;
    // vector to store the pixel coordinates of detected checker board corners 
    vector<Point2f> corner_pts;
    bool success;

    // Looping over all the images in the directory
    for(auto image: images) {
        frame = imread(image);
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Finding checker board corners
        // If desired number of corners are found in the image then success = true  
        success = findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        
        /* 
        * If desired number of corner are detected,
        * we refine the pixel coordinates and display 
        * them on the images of checker board
        */
        if(success) {
            TermCriteria criteria(TermCriteria::EPS | TermCriteria::COUNT, 30, 0.001);
            
            // refining pixel coordinates for given 2d points.
            cornerSubPix(gray, corner_pts, cv::Size(11,11), cv::Size(-1,-1), criteria);
            
            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }
    }

    Mat R, T;

    /*
    * Performing camera calibration by 
    * passing the value of known 3D points (objpoints)
    * and corresponding pixel coordinates of the 
    * detected corners (imgpoints)
    */
    calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T);
}

void drawCubeWireframe(InputOutputArray& image, InputArray& cameraMatrix, InputArray& distCoeffs, InputArray& rvec, InputArray& tvec, float l) {

    CV_Assert(
        image.getMat().total() != 0 &&
        (image.getMat().channels() == 1 || image.getMat().channels() == 3)
    );
    CV_Assert(l > 0);
    float half_l = l / 2.0;

    // project cube points
    vector<Point3f> axisPoints;
    axisPoints.push_back(Point3f(l, l, half_l));
    axisPoints.push_back(Point3f(0, l, half_l));
    axisPoints.push_back(Point3f(0, 0, half_l));
    axisPoints.push_back(Point3f(l, 0, half_l));
    axisPoints.push_back(Point3f(l, l, 0));
    axisPoints.push_back(Point3f(0, l, 0));
    axisPoints.push_back(Point3f(0, 0, 0));
    axisPoints.push_back(Point3f(l, 0, 0));

    vector<Point2f> imagePoints;
    projectPoints(
        axisPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints
    );

    // draw cube edges lines
    line(image, imagePoints[0], imagePoints[1], Scalar(255, 0, 0), 3);
    line(image, imagePoints[0], imagePoints[3], Scalar(255, 0, 0), 3);
    line(image, imagePoints[0], imagePoints[4], Scalar(255, 0, 0), 3);
    line(image, imagePoints[1], imagePoints[2], Scalar(255, 0, 0), 3);
    line(image, imagePoints[1], imagePoints[5], Scalar(255, 0, 0), 3);
    line(image, imagePoints[2], imagePoints[3], Scalar(255, 0, 0), 3);
    line(image, imagePoints[2], imagePoints[6], Scalar(255, 0, 0), 3);
    line(image, imagePoints[3], imagePoints[7], Scalar(255, 0, 0), 3);
    line(image, imagePoints[4], imagePoints[5], Scalar(255, 0, 0), 3);
    line(image, imagePoints[4], imagePoints[7], Scalar(255, 0, 0), 3);
    line(image, imagePoints[5], imagePoints[6], Scalar(255, 0, 0), 3);
    line(image, imagePoints[6], imagePoints[7], Scalar(255, 0, 0), 3);
}

void drawTrajectory(vector<Point2f>& trajectory, Mat& image) {
    for (auto point = trajectory.begin(); point < trajectory.end() - 1; ++point) {
            line(image, point[0], point[1], Scalar(0, 0, 255), 3, LINE_AA, 0);
    }
}

int main() {
    Mat trj_image = imread("map.jpg", IMREAD_COLOR);
    resize(trj_image, trj_image, Size(trj_image.cols / 2, trj_image.rows / 2));

    vector<Point2f> trajectory;
    Mat cameraMatrix, distCoeffs;
    getCameraParams(cameraMatrix, distCoeffs);
    VideoCapture cap("black.mp4");

    if (!cap.isOpened()) {
        return -1;
    }

    namedWindow("Detection", WindowFlags::WINDOW_AUTOSIZE);
    namedWindow("Trajectory", WindowFlags::WINDOW_AUTOSIZE);
    vector<vector<Point>> stickers;

    while (1) {
        Mat frame;
        bool bSuccess = cap.read(frame);
        if (!bSuccess) {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }

        QRCodeDetector qrDecoder = QRCodeDetector();
        
        Mat bbox, rectifiedImage;

        qrDecoder.detect(frame, bbox);
        std::string data = qrDecoder.detectAndDecode(frame, bbox, rectifiedImage);
    
        vector<Point3f> obj_coords;
        obj_coords.push_back(Point3f(0, 0, 0));
        obj_coords.push_back(Point3f(0, QR_CODE_WIDTH, 0));
        obj_coords.push_back(Point3f(QR_CODE_WIDTH, QR_CODE_WIDTH, 0));
        obj_coords.push_back(Point3f(QR_CODE_WIDTH, 0, 0));
        
        Mat R,T;

        if(!bbox.empty()) {
            solvePnP(obj_coords, bbox, cameraMatrix, distCoeffs, R, T);
            // aruco::drawAxis(frame, cameraMatrix, distCoeffs, R, T, 0.07);
            drawCubeWireframe(frame, cameraMatrix, distCoeffs, R, T, QR_CODE_WIDTH);
            Mat Rt, Rtt;
            Rodrigues(R,Rt);
            transpose(Rt,Rtt);
            cout << -Rtt*T << endl;
            Mat1f temp = -Rtt*T;
            Point2f point((temp(0) + QR_CENTER_X) * (trj_image.cols / MAP_WIDTH), (temp(1) + QR_CENTER_Y)*(trj_image.rows / MAP_HEIGHT));
            trajectory.push_back(point);
        }

        imshow("Trajectory", trj_image);
        drawTrajectory(trajectory, trj_image);

        imshow("Detection", frame);
        if (waitKey(30) == 27) {
            break;
        }
    }
    return 0;
}
