#include "hand_drawing.h"
#include <iostream>

HandDrawingInterface::HandDrawingInterface() {
    // Initialize drawing canvas (black background)
    drawing_canvas = cv::Mat::zeros(canvas_size, canvas_size, CV_8UC1);
    display_canvas = cv::Mat::zeros(canvas_size, canvas_size, CV_8UC3);
    prediction_display = cv::Mat::zeros(200, 400, CV_8UC3);
    
    // Create windows
    cv::namedWindow("Draw Your Digit", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Neural Network Prediction", cv::WINDOW_AUTOSIZE);
    
    show_instructions();
}

void HandDrawingInterface::set_network_params(const torch::Tensor& w1_param, const torch::Tensor& b1_param,
                                             const torch::Tensor& w2_param, const torch::Tensor& b2_param,
                                             const torch::Tensor& w3_param, const torch::Tensor& b3_param,
                                             const torch::Device& device_param) {
    w1 = w1_param;
    b1 = b1_param;
    w2 = w2_param;
    b2 = b2_param;
    w3 = w3_param;
    b3 = b3_param;
    device = device_param;
}

void HandDrawingInterface::start_drawing_session() {
    std::cout << "\\n=== Hand Drawing Recognition ===\\n";
    std::cout << "Draw digits with your mouse in the drawing window!\\n";
    std::cout << "Controls:\\n";
    std::cout << "  - Left click and drag to draw\\n";
    std::cout << "  - Press 'c' to clear canvas\\n";
    std::cout << "  - Press 'ESC' to exit\\n\\n";
    
    // Set mouse callback
    cv::setMouseCallback("Draw Your Digit", mouse_callback, this);
    
    while (true) {
        // Update display
        cv::cvtColor(drawing_canvas, display_canvas, cv::COLOR_GRAY2BGR);
        
        // Add grid lines for better drawing guidance
        for (int i = 0; i < canvas_size; i += 50) {
            cv::line(display_canvas, cv::Point(i, 0), cv::Point(i, canvas_size), cv::Scalar(30, 30, 30), 1);
            cv::line(display_canvas, cv::Point(0, i), cv::Point(canvas_size, i), cv::Scalar(30, 30, 30), 1);
        }
        
        cv::imshow("Draw Your Digit", display_canvas);
        
        // Get prediction if there's drawing
        cv::Scalar sum = cv::sum(drawing_canvas);
        if (sum[0] > 0) {  // If there's something drawn
            int prediction = predict_digit();
            // Note: confidence calculation would need the full forward pass
            update_prediction_display(prediction, 0.0f);
        } else {
            // Clear prediction display when canvas is empty
            prediction_display.setTo(cv::Scalar(40, 40, 40));
            cv::putText(prediction_display, "Draw a digit...", cv::Point(20, 100), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            cv::imshow("Neural Network Prediction", prediction_display);
        }
        
        int key = cv::waitKey(30) & 0xFF;
        if (key == 27) {  // ESC key
            break;
        } else if (key == 'c' || key == 'C') {
            clear_canvas();
        }
    }
    
    cv::destroyWindow("Draw Your Digit");
    cv::destroyWindow("Neural Network Prediction");
}

void HandDrawingInterface::mouse_callback(int event, int x, int y, int flags, void* userdata) {
    HandDrawingInterface* interface = static_cast<HandDrawingInterface*>(userdata);
    interface->handle_mouse(event, x, y, flags);
}

void HandDrawingInterface::handle_mouse(int event, int x, int y, int flags) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        is_drawing = true;
    } else if (event == cv::EVENT_LBUTTONUP) {
        is_drawing = false;
    }
    
    if (is_drawing && (flags & cv::EVENT_FLAG_LBUTTON)) {
        // Draw with a brush
        cv::circle(drawing_canvas, cv::Point(x, y), 12, cv::Scalar(255), -1);
        
        // Also draw connecting line for smooth strokes
        static cv::Point last_point(-1, -1);
        if (last_point.x >= 0 && last_point.y >= 0) {
            cv::line(drawing_canvas, last_point, cv::Point(x, y), cv::Scalar(255), 24);
        }
        last_point = cv::Point(x, y);
    } else {
        static cv::Point last_point(-1, -1);
        last_point = cv::Point(-1, -1);  // Reset for next stroke
    }
}

torch::Tensor HandDrawingInterface::preprocess_drawing() {
    // Convert drawing to 28x28 tensor for neural network
    cv::Mat resized;
    cv::resize(drawing_canvas, resized, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
    
    // Convert to float and normalize to [0, 1]
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0/255.0);
    
    // Convert to PyTorch tensor
    torch::Tensor tensor = torch::from_blob(float_img.data, {28, 28}, torch::kFloat);
    tensor = tensor.flatten();  // Convert to 784 dimensions
    tensor = tensor.unsqueeze(0);  // Add batch dimension [1, 784]
    
    return tensor.to(device);
}

int HandDrawingInterface::predict_digit() {
    torch::NoGradGuard no_grad;
    
    auto input = preprocess_drawing();
    
    // Forward pass through network
    auto z1 = torch::mm(input, w1) + b1;
    auto a1 = torch::relu(z1);
    auto z2 = torch::mm(a1, w2) + b2;
    auto a2 = torch::relu(z2);
    auto z3 = torch::mm(a2, w3) + b3;
    
    auto probs = torch::softmax(z3, 1);
    auto prediction = torch::argmax(probs, 1).item<int>();
    
    return prediction;
}

void HandDrawingInterface::update_prediction_display(int prediction, float confidence) {
    prediction_display.setTo(cv::Scalar(40, 40, 40));
    
    // Large prediction display
    std::string pred_text = std::to_string(prediction);
    cv::putText(prediction_display, pred_text, cv::Point(150, 120), 
               cv::FONT_HERSHEY_SIMPLEX, 4.0, cv::Scalar(0, 255, 0), 8);
    
    // Label
    cv::putText(prediction_display, "Prediction:", cv::Point(20, 40), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    
    // Instructions
    cv::putText(prediction_display, "Press 'c' to clear", cv::Point(20, 170), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
    
    cv::imshow("Neural Network Prediction", prediction_display);
}

void HandDrawingInterface::clear_canvas() {
    drawing_canvas.setTo(cv::Scalar(0));
}

void HandDrawingInterface::show_instructions() {
    std::cout << "Hand Drawing Interface initialized!\\n";
    std::cout << "Draw digits 0-9 and watch the neural network recognize them in real-time.\\n";
}
