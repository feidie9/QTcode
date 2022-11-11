#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



using namespace cv;
using namespace cv::dnn;
using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_pushButton_4_clicked();

private:
    Ui::MainWindow *ui;
    VideoCapture capture;
    Mat frame;    
    Mat gray, qrcode_bin;
    QRCodeDetector qrcodedetector;
    vector<Point> points;
    string information;
    bool isQRcode;
    std::string root_dir = "/usr/local/include/opencv4/opencv2/dnn/face_detector/";
    dnn::Net net = dnn::readNetFromTensorflow(root_dir+"opencv_face_detector_uint8.pb",root_dir+"opencv_face_detector.pbtxt");
};
#endif // MAINWINDOW_H
