#include "mnist_visualizer.h"

MNISTVisualizer::MNISTVisualizer() {
    cv::namedWindow("MNIST Classifier Visualizer", cv::WINDOW_AUTOSIZE);
    canvas = cv::Mat::zeros(window_height, window_width, CV_8UC3);
}
    
void MNISTVisualizer::show_digit(const torch::Tensor& digit_tensor, int prediction, float confidence, int actual) {
        canvas.setTo(cv::Scalar(40, 40, 40)); // Dark background
        
        // Convert tensor to OpenCV Mat (28x28)
        auto digit_cpu = digit_tensor.cpu();
        cv::Mat digit_img(28, 28, CV_32F, digit_cpu.data_ptr<float>());
        
        // Scale to 0-255 and convert to 8-bit
        cv::Mat digit_display;
        digit_img.convertTo(digit_display, CV_8U, 255.0);
        
        // Resize digit for display (280x280)
        cv::Mat digit_large;
        cv::resize(digit_display, digit_large, cv::Size(280, 280), 0, 0, cv::INTER_NEAREST);
        
        // Convert to 3-channel for display
        cv::Mat digit_color;
        cv::cvtColor(digit_large, digit_color, cv::COLOR_GRAY2BGR);
        
        // Place digit in canvas
        cv::Rect digit_roi(50, 50, 280, 280);
        digit_color.copyTo(canvas(digit_roi));
        
        // Add border
        cv::rectangle(canvas, digit_roi, cv::Scalar(255, 255, 255), 2);
        
        // Add prediction text
        std::string pred_text = "Predicted: " + std::to_string(prediction);
        cv::putText(canvas, pred_text, cv::Point(400, 100), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        // Add confidence
        std::string conf_text = "Confidence: " + std::to_string((int)(confidence * 100)) + "%";
        cv::putText(canvas, conf_text, cv::Point(400, 150), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        
        // Add actual label if provided
        if (actual >= 0) {
            std::string actual_text = "Actual: " + std::to_string(actual);
            cv::Scalar color = (actual == prediction) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::putText(canvas, actual_text, cv::Point(400, 200), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
        }
        
        cv::imshow("MNIST Classifier Visualizer", canvas);
        cv::waitKey(1); // Non-blocking update
    }
    
void MNISTVisualizer::show_training_progress(int epoch, float loss, float accuracy) {
        // Create progress display
        cv::Mat progress_canvas = cv::Mat::zeros(200, 600, CV_8UC3);
        
        std::string epoch_text = "Epoch: " + std::to_string(epoch);
        cv::putText(progress_canvas, epoch_text, cv::Point(20, 50), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        
        std::string loss_text = "Loss: " + std::to_string(loss);
        cv::putText(progress_canvas, loss_text, cv::Point(20, 100), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        
        std::string acc_text = "Accuracy: " + std::to_string((int)(accuracy * 100)) + "%";
        cv::putText(progress_canvas, acc_text, cv::Point(20, 150), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("Training Progress", progress_canvas);
        cv::waitKey(1);
    }
    
void MNISTVisualizer::show_predictions_grid(const std::vector<torch::Tensor>& digits, 
                              const std::vector<int>& predictions,
                              const std::vector<float>& confidences,
                              const std::vector<int>& actuals) {
        
        int grid_size = 3; // 3x3 grid
        int cell_size = 180;
        cv::Mat grid_canvas = cv::Mat::zeros(grid_size * cell_size + 50, 
                                           grid_size * cell_size + 50, CV_8UC3);
        
        for (int i = 0; i < std::min((int)digits.size(), 9); ++i) {
            int row = i / grid_size;
            int col = i % grid_size;
            
            // Convert tensor to image
            auto digit_cpu = digits[i].cpu();
            cv::Mat digit_img(28, 28, CV_32F, digit_cpu.data_ptr<float>());
            cv::Mat digit_display;
            digit_img.convertTo(digit_display, CV_8U, 255.0);
            
            // Resize and place
            cv::Mat digit_resized;
            cv::resize(digit_display, digit_resized, cv::Size(140, 140), 0, 0, cv::INTER_NEAREST);
            
            cv::Mat digit_color;
            cv::cvtColor(digit_resized, digit_color, cv::COLOR_GRAY2BGR);
            
            int x = col * cell_size + 20;
            int y = row * cell_size + 20;
            cv::Rect cell_roi(x, y, 140, 140);
            digit_color.copyTo(grid_canvas(cell_roi));
            
            // Add prediction text
            std::string pred = std::to_string(predictions[i]);
            cv::Scalar color = cv::Scalar(0, 255, 0);
            if (!actuals.empty() && actuals[i] != predictions[i]) {
                color = cv::Scalar(0, 0, 255);
            }
            
            cv::putText(grid_canvas, pred, cv::Point(x, y + 170), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
        }
        
        cv::imshow("Predictions Grid", grid_canvas);
        cv::waitKey(1);
    }
    
bool MNISTVisualizer::wait_for_key(int delay) {
        int key = cv::waitKey(delay);
        return key != -1;
    }
    
void MNISTVisualizer::close() {
    cv::destroyAllWindows();
}
