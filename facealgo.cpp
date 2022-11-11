#include <facealgo.h>

FaceAlgo::FaceAlgo(){
    std::cout<<"creat instance"<<std::endl;
}

void FaceAlgo::initFaceModels(std::string detect_model_path, std::string recog_model_path, std::string face_db_dir) {

    this->faceDetector = cv::FaceDetectorYN::create(detect_model_path, "", cv::Size(300, 300), 0.9f, 0.3f, 500);
    this->faceRecognizer = cv::FaceRecognizerSF::create(recog_model_path, "");
    std::vector<std::string> fileNames;
    cv::glob(face_db_dir, fileNames);
    for(std::string file_path : fileNames){
        cv::Mat image = cv::imread(file_path);
        int pos = static_cast<int>(file_path.find("\\"));
        std::string image_name = file_path.substr(pos+1, file_path.length() - pos - 5);
        this->registFace(image, image_name);
        std::cout<<"file name : " << image_name<< ".jpg"<<std::endl;
    }
}

void FaceAlgo::detectFace(cv::Mat &image, std::vector<std::shared_ptr<faceInfo>> &infoList, bool showFPS) {
    cv::TickMeter tm;
    std::string msg = "FPS: ";
    tm.start();
    // Set input size before inference
    this->faceDetector->setInputSize(image.size());

    // Inference
    cv::Mat faces;
    this->faceDetector->detect(image, faces);
    tm.stop();
    // Draw results on the input image
    int thickness = 2;
    for (int i = 0; i < faces.rows; i++)
    {
        // Draw bounding box
        auto fi = std::shared_ptr<faceInfo>(new faceInfo());
        fi->name = "Unknown";
        faces.row(0).copyTo(fi->detResult);
        infoList.push_back(fi);
    }
    if(showFPS) {
        putText(image, msg + std::to_string(tm.getFPS()), cv::Point(15, 25), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), thickness);
    }
}

void FaceAlgo::matchFace(cv::Mat &frame, std::vector<std::shared_ptr<faceInfo>> &infoList, bool l2) {
    double cosine_similar_thresh = 0.363;
    double l2norm_similar_thresh = 1.128;
    for(auto face : infoList) {
        cv::Mat aligned_face, feature;
        faceRecognizer->alignCrop(frame, face->detResult, aligned_face);
        faceRecognizer->feature(aligned_face, feature);
        double min_dist = 100.0;
        double max_cosine = 0.0;
        std::string matchedName = "Unknown";
        for(auto item : this->face_models) {
            //std::cout<<"face_models.item :" << item.first << std::endl;
            //std::cout<<"face_models.item :" << item.second << std::endl;
            if(l2) {
                double L2_score = faceRecognizer->match(feature, item.second, cv::FaceRecognizerSF::DisType::FR_NORM_L2);
                if(L2_score < min_dist) {
                    min_dist = L2_score;
                    matchedName = item.first;
                }
            } else {
                double cos_score = faceRecognizer->match(feature, item.second, cv::FaceRecognizerSF::DisType::FR_COSINE);
                if(cos_score > max_cosine) {
                    max_cosine = cos_score;
                    matchedName = item.first;
                }
            }
        }
        //std::cout<<"matchedName :" << matchedName << std::endl;
        if(max_cosine > cosine_similar_thresh) {
            face->name.clear();
            face->name.append(matchedName);
        }
        if(l2 && min_dist < l2norm_similar_thresh) {
            face->name.clear();
            face->name.append(matchedName);
        }
        // std::cout<<"face.name :" << face->name << std::endl;
        // std::cout<<"max_cosine :" << max_cosine<< std::endl;
        // std::cout<<"min_dist :" << min_dist<< std::endl;
    }
}

void FaceAlgo::registFace(cv::Mat &frame, std::string name) {
    this->faceDetector->setInputSize(frame.size());

    // Inference
    cv::Mat faces;
    this->faceDetector->detect(frame, faces);
    cv::Mat aligned_face, feature;
    faceRecognizer->alignCrop(frame, faces.row(0), aligned_face);
    faceRecognizer->feature(aligned_face, feature);
    this->face_models.insert(std::pair<std::string, cv::Mat>(name, feature.clone()));
}
