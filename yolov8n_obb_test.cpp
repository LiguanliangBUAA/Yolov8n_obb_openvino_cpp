#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

cv::Mat letterbox(const cv::Mat& source, int target_size, float& scale, float& pad_w, float& pad_h) {
    int h = source.rows;
    int w = source.cols;

    scale = std::min((float)target_size / w, (float)target_size / h);
    int new_w = std::round(w * scale);
    int new_h = std::round(h * scale);

    pad_w = (target_size - new_w) / 2.0f;
    pad_h = (target_size - new_h) / 2.0f;

    cv::Mat resized;
    cv::resize(source, resized, cv::Size(new_w, new_h));

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, int(pad_h), target_size - new_h - int(pad_h), 
                       int(pad_w), target_size - new_w - int(pad_w), 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return padded;
}

int main() {
    std::string model_xml = "../yolon1024HD-clean-AdamW_openvino_model/yolon1024HD-clean-AdamW.xml"; 
    std::string image_path = "../test_image.jpg";
    int target_size = 1024;
    float conf_threshold = 0.25f;
    float nms_threshold = 0.4f;
    std::vector<std::string> class_names = {"walls", "shadows", "columns", "others"};

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Unable to read image: " << image_path << std::endl;
        return -1;
    }

    // Preprocessing (Letterbox)
    float scale, pad_w, pad_h;
    cv::Mat input_image = letterbox(image, target_size, scale, pad_w, pad_h);
    
    // Convert to RGB (YOLO training usually uses RGB)
    cv::Mat rgb_image;
    cv::cvtColor(input_image, rgb_image, cv::COLOR_BGR2RGB);

    // OpenVINO Initialization and Inference
    std::cout << "Loading OpenVINO model..." << std::endl;
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model(model_xml, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Construct blob, note 1.0/255.0 normalization
    cv::Mat blob = cv::dnn::blobFromImage(rgb_image, 1.0 / 255.0, cv::Size(target_size, target_size), cv::Scalar(), false, false);
    
    ov::Tensor input_tensor(ov::element::f32, {1, 3, (size_t)target_size, (size_t)target_size}, blob.ptr<float>());
    infer_request.set_input_tensor(input_tensor);

    std::cout << "Performing inference..." << std::endl;
    infer_request.infer();

    // Parse output
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    const float* out_data = output_tensor.data<float>();
    auto shape = output_tensor.get_shape();

    // Determine dimension order
    int dim1 = shape[1];
    int dim2 = shape[2];
    int anchors = (dim1 > dim2) ? dim1 : dim2;
    int channels = (dim1 > dim2) ? dim2 : dim1;
    bool is_transposed = (dim1 > dim2);

    int num_classes = channels - 5; // cx, cy, w, h, angle

    std::cout << "Model channels: " << channels << ", Anchors: " << anchors << ", Parsed classes: " << num_classes << std::endl;
    if (num_classes != class_names.size()) {
        std::cerr << "[WARNING] Parsed class count (" << num_classes << ") does not match your specified count (" << class_names.size() << ")!" << std::endl;
    }

    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < anchors; ++i) {
        float max_score = 0.0f;
        int best_class_id = -1;
        float cx, cy, w, h, angle_rad;

        if (is_transposed) {
            const float* anchor_data = out_data + i * channels;
            for (int c = 0; c < num_classes; ++c) {
                if (anchor_data[4 + c] > max_score) {
                    max_score = anchor_data[4 + c];
                    best_class_id = c;
                }
            }
            cx = anchor_data[0];
            cy = anchor_data[1];
            w  = anchor_data[2];
            h  = anchor_data[3];
            angle_rad = anchor_data[channels - 1];
        } else {
            for (int c = 0; c < num_classes; ++c) {
                if (out_data[(4 + c) * anchors + i] > max_score) {
                    max_score = out_data[(4 + c) * anchors + i];
                    best_class_id = c;
                }
            }
            cx = out_data[0 * anchors + i];
            cy = out_data[1 * anchors + i];
            w  = out_data[2 * anchors + i];
            h  = out_data[3 * anchors + i];
            angle_rad = out_data[(channels - 1) * anchors + i];
        }

        if (max_score >= conf_threshold) {
            // Post-process to original image scale
            float orig_cx = (cx - pad_w) / scale;
            float orig_cy = (cy - pad_h) / scale;
            float orig_w  = w / scale;
            float orig_h  = h / scale;

            // YOLO OBB: radians, OpenCV RotatedRect: degrees
            float angle_deg = angle_rad * 180.0f / CV_PI;

            boxes.push_back(cv::RotatedRect(cv::Point2f(orig_cx, orig_cy), cv::Size2f(orig_w, orig_h), angle_deg));
            confidences.push_back(max_score);
            class_ids.push_back(best_class_id);
        }
    }

    std::cout << "Number of boxes before NMS: " << boxes.size() << std::endl;

    // NMS for rotated boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    
    std::cout << "Number of boxes after NMS: " << indices.size() << std::endl;

    // Draw results on the original image
    for (int idx : indices) {
        const auto& rect = boxes[idx];
        int cls_id = class_ids[idx];
        float conf = confidences[idx];
        
        cv::Point2f pts[4];
        rect.points(pts);

        // Draw the rotated rectangle
        for (int j = 0; j < 4; ++j) {
            cv::line(image, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }

        // Draw center point
        cv::circle(image, rect.center, 3, cv::Scalar(0, 0, 255), -1);

        // Draw label
        std::string label = class_names[cls_id] + " " + std::to_string(conf).substr(0, 4);
        cv::putText(image, label, pts[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
        
        std::cout << "Detected: " << class_names[cls_id] << " Confidence: " << conf 
                  << " Center: (" << rect.center.x << ", " << rect.center.y 
                  << ") Angle: " << rect.angle << std::endl;
    }

    // Save and display
    cv::imwrite("result.jpg", image);
    std::cout << "Result saved as result.jpg" << std::endl;

    return 0;
}