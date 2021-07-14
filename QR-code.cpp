#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

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

        std::string data = qrDecoder.detectAndDecode(frame, bbox, rectifiedImage);

        if (data.length() > 0)
        {
            cout << "Decoded Data : " << data << endl;

            display(frame, bbox);
            rectifiedImage.convertTo(rectifiedImage, CV_8UC3);
           // imshow("Detection", rectifiedImage);

            //waitKey(0);
        }
        else {
            cout << "QR Code not detected" << endl;
            imshow("Detection", frame);
        }

        if (waitKey(30) == 27)
        {
            break;
        }
    }
    return 0;
}
