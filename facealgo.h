#ifndef FACEALGO_H
#define FACEALGO_H

#endif // FACEALGO_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct faceInfo
{
    std::string name;
    cv::Mat detResult;
};

class FaceAlgo{
public:
    FaceAlgo();
    void initFaceModels(std::string detect_model_path,std::string recog_model_path,std::string face_db_dir);
    void detectFace(cv::Mat &farme,std::vector<std::shared_ptr<faceInfo>> &result, bool shoeFPS);
    void matchFace(cv::Mat &farme,std::vector<std::shared_ptr<faceInfo>> &result, bool l2=false);
    void registFace(cv::Mat &faceRoi,std::string name);

private:
    std::map<std::string, cv::Mat> face_models;
    cv::Ptr<cv::FaceDetectorYN> faceDetector;
    cv::Ptr<cv::FaceRecognizerSF> faceRecognizer;
};
