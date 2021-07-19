#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
// #include "opencv2/xfeatures2d.hpp"
// #include "opencv2/features2d.hpp"

using namespace cv;
using namespace std;


// using namespace cv::xfeatures2d;

// const int MAX_FEATURES = 500; const float GOOD_MATCH_PERCENT = 0.15f;

// void alignimages(Mat &im1, Mat &im2, Mat &im1Reg, Mat &h) {
//     Mat im1Gray, im2Gray;
//     cvtColor(im1, im1Gray, CV_BGR2GRAY); cvtColor(im2, im2Gray, CV_BGR2GRAY); // Convert images to grayscale

//     std::vector<KeyPoint> keypoints1, keypoints2; // Variables to store keypoints and descriptors

//     Mat descriptors1, descriptors2;

//     Ptr<Feature2D> orb = ORB::create(MAX_FEATURES); // Detect ORB features and compute descriptors.
//     orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);

//     orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

//     std::vector<DMatch> matches;

//     Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming"); // Match features
//     matcher->match(descriptors1, descriptors2, matches, Mat());

//     std::sort(matches.begin(), matches.end()); // Sort matches by score

//     const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT; // Remove not so good matches
//     matches.erase(matches.begin()+numGoodMatches, matches.end());

//     Mat imMatches;
//     drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches); // Draw top matches
//     imwrite("matches.jpg", imMatches);

//     std::vector<Point2f> points1, points2; // Extract location of good matches
//     for( size_t i = 0; i < matches.size(); i++ ){
//         points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
//         points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
//     }
//     h = findHomography( points1, points2, RANSAC ); // Find homography
//     warpPerspective(im1, im1Reg, h, im2.size()); // Use homography to warp image
// }

void find_corners(vector<Point> contour, Point points[4])
{
    Point &x_max=points[2], &x_min=points[0], 
            &y_max=points[3], &y_min=points[1];
    x_max = x_min = y_max = y_min = contour[0];
    for(int i=1; i<contour.size(); i++){
        Point cur = contour[i];
        if(x_max.x < cur.x) x_max = cur;
        if(x_min.x > cur.x) x_min = cur;
        if(y_max.y < cur.y) y_max = cur;
        if(y_min.y > cur.y) y_min = cur;
    }
}

void display(Mat &im, Mat &bbox)
{
    int n = bbox.rows;
    for (int i = 0; i < n; i++)
    {
        line(im, Point2i(bbox.at<float>(i, 0), bbox.at<float>(i, 1)), 
        Point2i(bbox.at<float>((i + 1) % n, 0), bbox.at<float>((i + 1) % n, 1)), Scalar(255, 0, 0), 3);
    }
    
    imshow("Detection", im);
}

int main()
{

    VideoCapture cap(1);
    if (!cap.isOpened())
    {
        return -1;
    }

    namedWindow("Detection", WINDOW_AUTOSIZE);
    std::vector<std::vector<cv::Point>> stickers;

    Mat lastframe;
    cap.read(lastframe);
    while (1)
    {
        Mat frame;
        bool bSuccess = cap.read(frame);
        if (!bSuccess)
        {
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
        
        vector<vector<Point>>contours;
        findContours(blured, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if(contours.size()){
            Rect max_br = boundingRect(contours[0]);
            uint num = 0;
            for(uint i=0; i < contours.size(); i++){
                Rect br = boundingRect(contours[i]);
                // if(br.area() > max_br.area()) max_br = br;
                if(br.area() > max_br.area()){
                    max_br = br;
                    num = i;
                }
            }
            // cout<< max_br << endl;
            // rectangle(frame, max_br, Scalar(0, 220, 0), 3);
            Point corners[4];
            find_corners(contours[num], corners);
            // line(frame, corners[0], corners[1], Scalar(250, 0, 0), 3);
            // line(frame, corners[1], corners[2], Scalar(250, 0, 0), 3);
            // line(frame, corners[2], corners[3], Scalar(250, 0, 0), 3);
            // line(frame, corners[3], corners[0], Scalar(250, 0, 0), 3);
            vector<Point> p, x;
            for(int i=0; i<4; i++)
                x.push_back(corners[i]);
            p.push_back(Point(50,250)); p.push_back(Point(50, 50)); 
            p.push_back(Point(250,50)); p.push_back(Point(250,250)); 
            Mat h = findHomography( x, p, RANSAC ), im; 
            warpPerspective(frame, im, h, im.size());
            // ---------------------------------------------------
        QRCodeDetector qrDecoder = QRCodeDetector();
        
        Mat bbox, rectifiedImage;
        
        std::string data = qrDecoder.detectAndDecode(im, bbox, rectifiedImage);

        if (data.length() > 0)
        {
            display(im, bbox);
            rectifiedImage.convertTo(rectifiedImage, CV_8UC3);
        }
        else {
            cout << "QR Code not detected" << endl;
            imshow("Detection", im);
        }
            // ----------------------------------------------------
            imshow("xxxx", im);
        }
        // if(contours.size()){
        //     RotatedRect max_br = minAreaRect(contours[0]);
        //     for(uint i=0; i < contours.size(); i++){
        //         RotatedRect br = minAreaRect(contours[i]);
        //         if(br.size.area() > max_br.size.area()) max_br = br;
        //     }
        //     // cout<< max_br << endl;
        //     // rectangle(frame, max_br, Scalar(250, 0, 0), 3);
        //     Point2f points[4];
        //     max_br.points(points);
        //     line(frame, points[0], points[1], Scalar(250, 0, 0), 3);
        //     line(frame, points[1], points[2], Scalar(250, 0, 0), 3);
        //     line(frame, points[2], points[3], Scalar(250, 0, 0), 3);
        //     line(frame, points[3], points[0], Scalar(250, 0, 0), 3);
            
        // }
        
        // Mat imReg, homo;
        // alignimages(lastframe, frame, imReg, homo);
        // imshow("??", imReg);

        // h = findHomography( points1, points2, RANSAC ); 
        // warpPerspective(im1, im1Reg, h, im2.size());
        imshow("kekw", frame);
        imshow("Detection", blured);
        
        if (waitKey(30) == 27)
        {
            break;
        }
    }
    return 0;
}
