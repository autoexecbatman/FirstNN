#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>

class HandDrawingInterface {
private:
    cv::Mat drawing_canvas;
    cv::Mat display_canvas;
    cv::Mat prediction_display;
    int canvas_size = 400;
    bool is_drawing = false;
    
    // Neural network parameters (will be set from main)
    torch::Tensor w1, b1, w2, b2, w3, b3;
    torch::Device device = torch::kCPU;
    
public:
    HandDrawingInterface();
    
    void set_network_params(const torch::Tensor& w1, const torch::Tensor& b1,
                           const torch::Tensor& w2, const torch::Tensor& b2,
                           const torch::Tensor& w3, const torch::Tensor& b3,
                           const torch::Device& device);
    
    void start_drawing_session();
    
private:
    static void mouse_callback(int event, int x, int y, int flags, void* userdata);
    void handle_mouse(int event, int x, int y, int flags);
    
    torch::Tensor preprocess_drawing();
    int predict_digit();
    void update_prediction_display(int prediction, float confidence);
    void clear_canvas();
    void show_instructions();
};
