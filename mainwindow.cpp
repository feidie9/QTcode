#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    capture.set(CAP_PROP_POS_FRAMES,10);
    capture.set(cv::CAP_PROP_FPS,25.0);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT,320);
    capture.set(cv::CAP_PROP_FRAME_WIDTH,640);

}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_clicked()
{
    capture.open(0,cv::CAP_ANY);
    //capture.open(0);
    //capture.set(CAP_PROP_POS_FRAMES,10);
    capture.set(cv::CAP_PROP_FPS,25.0);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT,320);
    capture.set(cv::CAP_PROP_FRAME_WIDTH,640);
    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
    int frame_fps = capture.get(CAP_PROP_FPS);
    namedWindow("Face",WINDOW_FREERATIO);
    while (true){

        capture.read(frame);
        Mat blob = dnn::blobFromImage(frame,1.0,Size(300,300),Scalar(104,177,123),false,false);
        net.setInput(blob);
        Mat probs = net.forward();
        Mat detectorMat(probs.size[2],probs.size[3],CV_32F,probs.ptr<float>());
        for(int i=0;i<detectorMat.rows;i++){
           float confidence = detectorMat.at<float>(i,2);
           if (confidence>0.5) {
               int x1 = static_cast<int>(detectorMat.at<float>(i,3)*frame.cols);
               int y1 = static_cast<int>(detectorMat.at<float>(i,4)*frame.rows);
               int x2 = static_cast<int>(detectorMat.at<float>(i,5)*frame.cols);
               int y2 = static_cast<int>(detectorMat.at<float>(i,6)*frame.rows);
               Rect box(x1,y1,x2-y1,y2-y1);
               rectangle(frame,box,Scalar(0,255,255),2,8,0);
           }
        }
        cv::imshow("Face",frame);

        waitKey(10);
        std::cout<<"FPS:"<<frame_fps<<std::endl;
        std::cout<<"frame_width:"<<frame_width<<std::endl;
        std::cout<<"frame_height:"<<frame_height<<std::endl;
    }

}

void MainWindow::on_pushButton_2_clicked()
{
    capture.release();
}

void MainWindow::on_pushButton_3_clicked()
{  
   capture.open(0,cv::CAP_ANY);
   capture.set(cv::CAP_PROP_FPS,25.0);
   capture.set(cv::CAP_PROP_FRAME_HEIGHT,320);
   capture.set(cv::CAP_PROP_FRAME_WIDTH,640);
   namedWindow("Face",WINDOW_FREERATIO);
   while (true){

       capture.read(frame);
       cv::imshow("Face",frame);

       waitKey(10);
     }
}

void MainWindow::on_pushButton_4_clicked()
{
    capture.open(0,cv::CAP_ANY);
    capture.set(cv::CAP_PROP_FPS,25.0);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT,320);
    capture.set(cv::CAP_PROP_FRAME_WIDTH,640);
    namedWindow("Face",WINDOW_FREERATIO);
    while (true){

        capture.read(frame);
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        isQRcode = qrcodedetector.detect(gray, points);
        if (isQRcode)
            {
                //½âÂë¶þÎ¬Âë
                information = qrcodedetector.decode(gray, points, qrcode_bin);
                cout << points << endl;  //Êä³ö¶þÎ¬ÂëËÄ¸ö¶¥µãµÄ×ø±ê
            }

            //»æÖÆ¶þÎ¬ÂëµÄ±ß¿ò
            for (int i = 0; i < points.size(); i++)
            {
                if (i == points.size() - 1)
                {
                    line(frame, points[i], points[0], Scalar(0, 0, 255), 2, 8);
                    break;
                }
                line(frame, points[i], points[i + 1], Scalar(0, 0, 255), 2, 8);
            }
            //½«½âÂëÄÚÈÝÊä³öµ½Í¼Æ¬ÉÏ
            putText(frame, information.c_str(), Point(20, 30), 0, 1.0, Scalar(0, 0, 255), 2, 8);
            cv::imshow("Face",frame);
        waitKey(10);
      }

}
