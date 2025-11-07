#pragma once
#include "../neural_network/utils/Tensor.hpp"
#include "../neural_network/layers/ConvolutionLayer.hpp"
#include "../neural_network/layers/PoolingLayers.hpp"
#include "../neural_network/layers/FullyConnectedLayer.hpp"
#include "../neural_network/layers/ActivationFunctions.hpp"
#include "../neural_network/utils/LossFunctions.hpp"
#include "MedicalImageProcessor.hpp"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>

class MedicalCNN {
public:
    struct MedicalPrediction {
        double tumor_probability;
        double normal_probability;
        int predicted_class;  // 0 = normal, 1 = tumor
        double confidence;
        std::string diagnosis;
    };
    
    struct TrainingMetrics {
        double accuracy;
        double sensitivity;  // True Positive Rate (recall)
        double specificity;  // True Negative Rate
        double precision;
        double f1_score;
        int true_positives;
        int true_negatives;
        int false_positives;
        int false_negatives;
    };

private:
    // CNN Architecture Layers (LeNet-like for medical imaging)
    std::unique_ptr<ConvolutionLayer> conv1;     // 8 filters, 3x3
    std::unique_ptr<ConvolutionLayer> conv2;     // 16 filters, 3x3
    std::unique_ptr<MaxPoolingLayer> pool1;      // 2x2
    std::unique_ptr<MaxPoolingLayer> pool2;      // 2x2
    std::unique_ptr<FullyConnectedLayer> fc1;    // 128 neurons
    std::unique_ptr<FullyConnectedLayer> fc2;    // 2 neurons (normal/tumor)
    
    // Activation functions
    ActivationFunctions::ReLU relu;
    ActivationFunctions::Softmax softmax;
    
    // Layer outputs for visualization
    Tensor conv1_output;
    Tensor conv2_output;
    Tensor pool1_output;
    Tensor pool2_output;
    Tensor fc1_output;
    Tensor final_output;
    
    int input_channels;
    int input_height;
    int input_width;
    bool verbose;
    
public:
    MedicalCNN(int channels = 1, int height = 64, int width = 64, bool verb = false) 
        : input_channels(channels), input_height(height), input_width(width), verbose(verb) {
        
        initialize_architecture();
        
        if (verbose) {
            print_architecture_summary();
        }
    }
    
    // Forward pass through the medical CNN
    MedicalPrediction predict(const Tensor& mri_input) {
        if (verbose) {
            std::cout << "\n=== Medical CNN Forward Pass ===" << std::endl;
            std::cout << "Input MRI shape: [" << mri_input.get_shape()[0] 
                      << ", " << mri_input.get_shape()[1] 
                      << ", " << mri_input.get_shape()[2] << "]" << std::endl;
        }
        
        // Layer 1: Conv2D + ReLU + MaxPool
        conv1_output = conv1->forward(mri_input);
        Tensor conv1_relu = relu.forward(conv1_output);
        pool1_output = pool1->forward(conv1_relu);
        
        if (verbose) {
            std::cout << "After Conv1+ReLU+Pool1: [" << pool1_output.get_shape()[0] 
                      << ", " << pool1_output.get_shape()[1] 
                      << ", " << pool1_output.get_shape()[2] << "]" << std::endl;
        }
        
        // Layer 2: Conv2D + ReLU + MaxPool
        conv2_output = conv2->forward(pool1_output);
        Tensor conv2_relu = relu.forward(conv2_output);
        pool2_output = pool2->forward(conv2_relu);
        
        if (verbose) {
            std::cout << "After Conv2+ReLU+Pool2: [" << pool2_output.get_shape()[0] 
                      << ", " << pool2_output.get_shape()[1] 
                      << ", " << pool2_output.get_shape()[2] << "]" << std::endl;
        }
        
        // Flatten for fully connected layers
        Tensor flattened = flatten_tensor(pool2_output);
        
        if (verbose) {
            std::cout << "After flattening: [" << flattened.get_shape()[0] << "]" << std::endl;
        }
        
        // Layer 3: Fully Connected + ReLU
        fc1_output = fc1->forward(flattened);
        Tensor fc1_relu = relu.forward(fc1_output);
        
        if (verbose) {
            std::cout << "After FC1+ReLU: [" << fc1_relu.get_shape()[0] << "]" << std::endl;
        }
        
        // Layer 4: Final Classification Layer
        Tensor fc2_output = fc2->forward(fc1_relu);
        final_output = softmax.forward(fc2_output);
        
        if (verbose) {
            std::cout << "Final classification: [" << final_output.get_shape()[0] << "]" << std::endl;
        }
        
        // Create medical prediction
        MedicalPrediction prediction;
        prediction.normal_probability = final_output[0];
        prediction.tumor_probability = final_output[1];
        prediction.predicted_class = (prediction.tumor_probability > prediction.normal_probability) ? 1 : 0;
        prediction.confidence = std::max(prediction.normal_probability, prediction.tumor_probability);
        
        // Generate clinical diagnosis
        if (prediction.predicted_class == 1) {
            if (prediction.confidence > 0.9) {
                prediction.diagnosis = "HIGH CONFIDENCE: Tumor detected - immediate further evaluation recommended";
            } else if (prediction.confidence > 0.7) {
                prediction.diagnosis = "MODERATE CONFIDENCE: Potential tumor - additional imaging advised";
            } else {
                prediction.diagnosis = "LOW CONFIDENCE: Possible abnormality - monitoring recommended";
            }
        } else {
            if (prediction.confidence > 0.9) {
                prediction.diagnosis = "NORMAL: No significant abnormalities detected";
            } else {
                prediction.diagnosis = "LIKELY NORMAL: Low probability of tumor";
            }
        }
        
        if (verbose) {
            std::cout << "\nMedical Assessment:" << std::endl;
            std::cout << "- Normal probability: " << std::fixed << std::setprecision(4) 
                      << prediction.normal_probability << std::endl;
            std::cout << "- Tumor probability: " << std::fixed << std::setprecision(4) 
                      << prediction.tumor_probability << std::endl;
            std::cout << "- Diagnosis: " << prediction.diagnosis << std::endl;
        }
        
        return prediction;
    }
    
    // Calculate medical evaluation metrics
    TrainingMetrics evaluate_medical_performance(const std::vector<Tensor>& test_images, 
                                               const std::vector<int>& true_labels) {
        TrainingMetrics metrics = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        
        for (size_t i = 0; i < test_images.size(); ++i) {
            MedicalPrediction pred = predict(test_images[i]);
            int predicted = pred.predicted_class;
            int actual = true_labels[i];
            
            if (actual == 1 && predicted == 1) {
                metrics.true_positives++;
            } else if (actual == 0 && predicted == 0) {
                metrics.true_negatives++;
            } else if (actual == 0 && predicted == 1) {
                metrics.false_positives++;
            } else if (actual == 1 && predicted == 0) {
                metrics.false_negatives++;
            }
        }
        
        int total = test_images.size();
        metrics.accuracy = (double)(metrics.true_positives + metrics.true_negatives) / total;
        
        // Sensitivity (Recall): TP / (TP + FN) - How well we detect actual tumors
        if (metrics.true_positives + metrics.false_negatives > 0) {
            metrics.sensitivity = (double)metrics.true_positives / 
                                (metrics.true_positives + metrics.false_negatives);
        }
        
        // Specificity: TN / (TN + FP) - How well we avoid false alarms
        if (metrics.true_negatives + metrics.false_positives > 0) {
            metrics.specificity = (double)metrics.true_negatives / 
                                (metrics.true_negatives + metrics.false_positives);
        }
        
        // Precision: TP / (TP + FP) - How many detected tumors are real
        if (metrics.true_positives + metrics.false_positives > 0) {
            metrics.precision = (double)metrics.true_positives / 
                              (metrics.true_positives + metrics.false_positives);
        }
        
        // F1 Score: Harmonic mean of precision and sensitivity
        if (metrics.precision + metrics.sensitivity > 0) {
            metrics.f1_score = 2 * (metrics.precision * metrics.sensitivity) / 
                             (metrics.precision + metrics.sensitivity);
        }
        
        return metrics;
    }
    
    // Print detailed medical evaluation report
    void print_medical_evaluation(const TrainingMetrics& metrics) {
        std::cout << "\n=== MEDICAL CNN EVALUATION REPORT ===" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        std::cout << "CONFUSION MATRIX:" << std::endl;
        std::cout << "                 Predicted" << std::endl;
        std::cout << "                 Normal  Tumor" << std::endl;
        std::cout << "Actual Normal  |   " << std::setw(3) << metrics.true_negatives 
                  << "     " << std::setw(3) << metrics.false_positives << std::endl;
        std::cout << "Actual Tumor   |   " << std::setw(3) << metrics.false_negatives 
                  << "     " << std::setw(3) << metrics.true_positives << std::endl;
        
        std::cout << "\nCLINICAL PERFORMANCE METRICS:" << std::endl;
        std::cout << "- Overall Accuracy: " << std::fixed << std::setprecision(3) 
                  << (metrics.accuracy * 100) << "%" << std::endl;
        std::cout << "- Sensitivity (Tumor Detection Rate): " << std::fixed << std::setprecision(3) 
                  << (metrics.sensitivity * 100) << "%" << std::endl;
        std::cout << "- Specificity (Normal Classification Rate): " << std::fixed << std::setprecision(3) 
                  << (metrics.specificity * 100) << "%" << std::endl;
        std::cout << "- Precision (Tumor Prediction Accuracy): " << std::fixed << std::setprecision(3) 
                  << (metrics.precision * 100) << "%" << std::endl;
        std::cout << "- F1 Score (Balanced Performance): " << std::fixed << std::setprecision(3) 
                  << (metrics.f1_score * 100) << "%" << std::endl;
        
        std::cout << "\nCLINICAL INTERPRETATION:" << std::endl;
        if (metrics.sensitivity >= 0.9) {
            std::cout << "✓ EXCELLENT tumor detection capability" << std::endl;
        } else if (metrics.sensitivity >= 0.8) {
            std::cout << "✓ GOOD tumor detection capability" << std::endl;
        } else {
            std::cout << "⚠ NEEDS IMPROVEMENT in tumor detection" << std::endl;
        }
        
        if (metrics.specificity >= 0.9) {
            std::cout << "✓ EXCELLENT at avoiding false alarms" << std::endl;
        } else if (metrics.specificity >= 0.8) {
            std::cout << "✓ GOOD at avoiding false alarms" << std::endl;
        } else {
            std::cout << "⚠ NEEDS IMPROVEMENT in reducing false positives" << std::endl;
        }
        
        std::cout << std::string(50, '=') << std::endl;
    }
    
    // Create activation heatmap for medical interpretation
    Tensor get_activation_heatmap() {
        // Use the last convolutional layer output (conv2) for localization
        if (conv2_output.get_shape().empty()) {
            std::cout << "Warning: No forward pass completed yet" << std::endl;
            return Tensor({1, 1, 1}, false);
        }
        
        // Average across all channels to create a single heatmap
        auto shape = conv2_output.get_shape();
        Tensor heatmap({1, shape[1], shape[2]}, false);
        
        for (int i = 0; i < shape[1]; ++i) {
            for (int j = 0; j < shape[2]; ++j) {
                double avg_activation = 0.0;
                for (int c = 0; c < shape[0]; ++c) {
                    avg_activation += std::abs(conv2_output(c, i, j));
                }
                avg_activation /= shape[0];
                heatmap(0, i, j) = avg_activation;
            }
        }
        
        return heatmap;
    }
    
    void set_verbose(bool verb) { verbose = verb; }

private:
    void initialize_architecture() {
        // Layer 1: Convolution 1 -> 8 filters, 3x3 kernel
        conv1 = std::make_unique<ConvolutionLayer>(input_channels, 8, 3, 1, 1);
        
        // Pool 1: MaxPool 2x2
        pool1 = std::make_unique<MaxPoolingLayer>(2, 2);
        
        // Layer 2: Convolution 8 -> 16 filters, 3x3 kernel
        conv2 = std::make_unique<ConvolutionLayer>(8, 16, 3, 1, 1);
        
        // Pool 2: MaxPool 2x2
        pool2 = std::make_unique<MaxPoolingLayer>(2, 2);
        
        // Calculate flattened size after convolutions and pooling
        // Input: 64x64 -> Conv1+Pool1: 32x32 -> Conv2+Pool2: 16x16
        // Flattened size: 16 channels * 16 * 16 = 4096
        int flattened_size = 16 * (input_height / 4) * (input_width / 4);
        
        // Layer 3: Fully Connected 4096 -> 128
        fc1 = std::make_unique<FullyConnectedLayer>(flattened_size, 128, true);
        
        // Layer 4: Final Classification 128 -> 2 (normal/tumor)
        fc2 = std::make_unique<FullyConnectedLayer>(128, 2, true);
    }
    
    void print_architecture_summary() {
        std::cout << "\n=== MEDICAL CNN ARCHITECTURE ===" << std::endl;
        std::cout << "Designed for brain tumor detection in MRI slices" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        std::cout << "Layer 1: Conv2D(1->8, 3x3) + ReLU + MaxPool(2x2)" << std::endl;
        std::cout << "         Output: [8, 32, 32]" << std::endl;
        
        std::cout << "Layer 2: Conv2D(8->16, 3x3) + ReLU + MaxPool(2x2)" << std::endl;
        std::cout << "         Output: [16, 16, 16]" << std::endl;
        
        std::cout << "Layer 3: Flatten + FC(4096->128) + ReLU" << std::endl;
        std::cout << "         Output: [128]" << std::endl;
        
        std::cout << "Layer 4: FC(128->2) + Softmax" << std::endl;
        std::cout << "         Output: [2] (Normal/Tumor probabilities)" << std::endl;
        
        // Calculate total parameters
        int conv1_params = 8 * (input_channels * 3 * 3 + 1);  // weights + bias
        int conv2_params = 16 * (8 * 3 * 3 + 1);
        int fc1_params = 128 * (4096 + 1);
        int fc2_params = 2 * (128 + 1);
        int total_params = conv1_params + conv2_params + fc1_params + fc2_params;
        
        std::cout << "\nParameter Count:" << std::endl;
        std::cout << "- Conv1: " << conv1_params << " parameters" << std::endl;
        std::cout << "- Conv2: " << conv2_params << " parameters" << std::endl;
        std::cout << "- FC1: " << fc1_params << " parameters" << std::endl;
        std::cout << "- FC2: " << fc2_params << " parameters" << std::endl;
        std::cout << "- TOTAL: " << total_params << " parameters" << std::endl;
        
        std::cout << "\nMedical Application: Brain tumor detection and localization" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
    }
    
    Tensor flatten_tensor(const Tensor& input) {
        auto shape = input.get_shape();
        int total_size = 1;
        for (int dim : shape) {
            total_size *= dim;
        }
        
        Tensor flattened({total_size}, false);
        int idx = 0;
        
        for (int c = 0; c < shape[0]; ++c) {
            for (int h = 0; h < shape[1]; ++h) {
                for (int w = 0; w < shape[2]; ++w) {
                    flattened[idx++] = input(c, h, w);
                }
            }
        }
        
        return flattened;
    }
};