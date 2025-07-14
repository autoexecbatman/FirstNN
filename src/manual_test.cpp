#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "=== Manual Binary Classifier Test ===" << std::endl;
    
    // Load the trained model weights (you'll need to save them first)
    // For now, let's create a simple manual test interface
    
    float a, b, c;
    char choice = 'y';
    
    while (choice == 'y' || choice == 'Y') {
        std::cout << "\nEnter three numbers (0-1): ";
        std::cin >> a >> b >> c;
        
        float sum = a + b + c;
        bool actual = sum > 1.5;
        
        std::cout << "You entered: [" << a << ", " << b << ", " << c << "]" << std::endl;
        std::cout << "Sum: " << sum << std::endl;
        std::cout << "Should be: " << (actual ? "1 (> 1.5)" : "0 (<= 1.5)") << std::endl;
        
        // Manual prediction logic (simple threshold for testing)
        std::cout << "Expected network output: " << (actual ? "~1.0" : "~0.0") << std::endl;
        
        std::cout << "\nTest another? (y/n): ";
        std::cin >> choice;
    }
    
    std::cout << "Manual testing complete!" << std::endl;
    return 0;
}