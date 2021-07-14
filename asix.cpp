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
void get_camera_params(Mat &cameraMatrix, Mat &distCoeffs)
{
    // Defining the dimensions of checkerboard
    int CHECKERBOARD[2]{7,7}; 
     // Creating vector to store vectors of 3D points for each checkerboard image
    std::vector<std::vector<cv::Point3f> > objpoints;

    // Creating vector to store vectors of 2D points for each checkerboard image
    std::vector<std::vector<cv::Point2f> > imgpoints;

    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for(int i{0}; i<CHECKERBOARD[1]; i++)
    {
        for(int j{0}; j<CHECKERBOARD[0]; j++)
        objp.push_back(cv::Point3f(j,i,0));
    }


    // Extracting path of individual image stored in a given directory
    std::vector<cv::String> images;
    // Path of the folder containing checkerboard images
    std::string path = "./images/";

    cv::glob(path, images);

    cv::Mat frame, gray;
    // vector to store the pixel coordinates of detected checker board corners 
    std::vector<cv::Point2f> corner_pts;
    bool success;

    // Looping over all the images in the directory
    for(int i{0}; i<images.size(); i++)
    {
        frame = cv::imread(images[i]);
        cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

        // Finding checker board corners
        // If desired number of corners are found in the image then success = true  
        success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        
        /* 
        * If desired number of corner are detected,
        * we refine the pixel coordinates and display 
        * them on the images of checker board
        */
        if(success)
        {
            cv::TermCriteria criteria(TermCriteria::EPS | TermCriteria::COUNT, 30, 0.001);
            
            // refining pixel coordinates for given 2d points.
            cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);
            
//          // Displaying the detected corner points on the checker board
//         cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
            
            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }

//        cv::imshow("Image",frame);
//        cv::waitKey(0);
    }

//    cv::destroyAllWindows();

   cv::Mat R, T;

    /*
    * Performing camera calibration by 
    * passing the value of known 3D points (objpoints)
    * and corresponding pixel coordinates of the 
    * detected corners (imgpoints)
    */
    cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T);
}

// void display(Mat &im, Mat &bbox)
// {
//     int n = bbox.rows;
//     for (int i = 0; i < n; i++)
//     {
//         line(im, Point2i(bbox.at<float>(i, 0), bbox.at<float>(i, 1)), 
//         Point2i(bbox.at<float>((i + 1) % n, 0), bbox.at<float>((i + 1) % n, 1)), Scalar(255, 0, 0), 3);
//     }
    
//     imshow("Detection", im);
// }

void drawCubeWireframe(
    cv::InputOutputArray image, cv::InputArray cameraMatrix,
    cv::InputArray distCoeffs, cv::InputArray rvec, cv::InputArray tvec,
    float l
)
{

    CV_Assert(
        image.getMat().total() != 0 &&
        (image.getMat().channels() == 1 || image.getMat().channels() == 3)
    );
    CV_Assert(l > 0);
    float half_l = l / 2.0;

    // project cube points
    std::vector<cv::Point3f> axisPoints;
    axisPoints.push_back(cv::Point3f(l, l, half_l));
    axisPoints.push_back(cv::Point3f(0, l, half_l));
    axisPoints.push_back(cv::Point3f(0, 0, half_l));
    axisPoints.push_back(cv::Point3f(l, 0, half_l));
    axisPoints.push_back(cv::Point3f(l, l, 0));
    axisPoints.push_back(cv::Point3f(0, l, 0));
    axisPoints.push_back(cv::Point3f(0, 0, 0));
    axisPoints.push_back(cv::Point3f(l, 0, 0));

    std::vector<cv::Point2f> imagePoints;
    projectPoints(
        axisPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints
    );

    // draw cube edges lines
    cv::line(image, imagePoints[0], imagePoints[1], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[0], imagePoints[4], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[1], imagePoints[2], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[1], imagePoints[5], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[2], imagePoints[3], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[2], imagePoints[6], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[3], imagePoints[7], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[4], imagePoints[5], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[4], imagePoints[7], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[5], imagePoints[6], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[6], imagePoints[7], cv::Scalar(255, 0, 0), 3);
}

void detectBlackRectangles(Mat& image, std::vector<std::vector<cv::Point>>& stickers) {
    cv::Mat image_hsv;
    std::vector< std::vector<cv::Point> > contours;
    cv::cvtColor(image, image_hsv, cv::COLOR_BGR2HSV ); 
    cv::Mat tmp_img(image.size(), CV_8U);

    cv::inRange(image_hsv, cv::Scalar(0, 0, 0, 0), cv::Scalar(180, 255, 30, 0), tmp_img);

    cv::dilate(tmp_img,tmp_img,cv::Mat(), cv::Point(-1,-1), 3);
    cv::erode(tmp_img,tmp_img,cv::Mat(), cv::Point(-1,-1), 1);
    cv::findContours(tmp_img, stickers, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    for (uint i = 0; i<contours.size(); i++) {
        cv::Mat sticker;
        cv::Rect rect=cv::boundingRect(contours[i]);
        image(rect).copyTo(sticker);
        cv::rectangle(image, rect, cv::Scalar(0, 250, 0), 2);
        stickers.push_back(sticker); 
    }
}

bool cmpPointX(const cv::Point &a, const cv::Point &b) {
    return a.x < b.x;
}

bool cmpPointY(const cv::Point &a, const cv::Point &b) {
    return a.y < b.y;
}

static bool stickers_comp(std::vector<cv::Point>& st1, std::vector<cv::Point>& st2) {
    int first_x = abs((*min_element(st1.begin(), st1.end(), cmpPointX)).x - (*max_element(st1.begin(), st1.end(), cmpPointX)).x);
    int first_y = abs((*min_element(st1.begin(), st1.end(), cmpPointY)).y - (*max_element(st1.begin(), st1.end(), cmpPointY)).y);

    int sec_x = abs((*min_element(st2.begin(), st2.end(), cmpPointX)).x - (*max_element(st2.begin(), st2.end(), cmpPointX)).x);
    int sec_y = abs((*min_element(st2.begin(), st2.end(), cmpPointY)).y - (*max_element(st2.begin(), st2.end(), cmpPointY)).y);

    return (first_x * first_y < sec_x * sec_y);
}


int main()
{
    Mat cameraMatrix, distCoeffs;
    get_camera_params(cameraMatrix, distCoeffs);
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        return -1;
    }

    namedWindow("Detection", cv::WindowFlags::WINDOW_AUTOSIZE);
    std::vector<std::vector<cv::Point>> stickers;

    while (1)
    {
        Mat frame;
        bool bSuccess = cap.read(frame);
        if (!bSuccess)
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }

        QRCodeDetector qrDecoder = QRCodeDetector();
        
        Mat bbox, rectifiedImage;

        qrDecoder.detect(frame, bbox);
        std::string data = qrDecoder.detectAndDecode(frame, bbox, rectifiedImage);

        if (data.length() > 0) {
            cout << "Decoded Data : " << data << endl;
        //     display(frame, bbox);
        //     rectifiedImage.convertTo(rectifiedImage, CV_8UC3);
        //     // imshow("Detection", rectifiedImage);

        //     // waitKey(0);
        }
        //  else {
        //     cout << "QR Code not detected" << endl;
        //     imshow("Detection", frame);
        // }
    
        detectBlackRectangles(frame, stickers);
        auto st = max_element(stickers.begin(), stickers.end(), stickers_comp);

        cv::Point sticker1; 
        sticker1.x = (*min_element((*st).begin(), (*st).end(), cmpPointX)).x;
        sticker1.y = (*min_element((*st).begin(), (*st).end(), cmpPointY)).y;
        cv::Point sticker2; 
        sticker2.x = (*max_element((*st).begin(), (*st).end(), cmpPointX)).x;
        sticker2.y = (*max_element((*st).begin(), (*st).end(), cmpPointY)).y;

        cv::rectangle(frame, Rect(sticker1, sticker2),cv::Scalar(0,250,0), 2);

        vector<Point3f> obj_coords;
        obj_coords.push_back(Point3f(0,0,0));
        obj_coords.push_back(Point3f(0,QR_CODE_WIDTH,0));
        obj_coords.push_back(Point3f(QR_CODE_WIDTH,QR_CODE_WIDTH,0));
        obj_coords.push_back(Point3f(QR_CODE_WIDTH,0,0));
        
        
        Mat R,T;

        if(!bbox.empty()) {
            solvePnP(obj_coords, bbox, cameraMatrix, distCoeffs, R, T);
            // cv::aruco::drawAxis(frame, cameraMatrix, distCoeffs, R, T, 0.07);
            drawCubeWireframe(frame, cameraMatrix, distCoeffs, R, T, QR_CODE_WIDTH);
        }

        imshow("Detection", frame);
        if (waitKey(30) == 27) {
            break;
        }
    }
    return 0;
}
