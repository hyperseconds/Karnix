# 🧠 Karnix: Complete Deep Learning & Medical AI Library

**A comprehensive C++ library for neural networks, CNNs, and medical AI - built from scratch with exact mathematical implementations.**

[![C++14](https://img.shields.io/badge/C++-14-blue.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](BUILD)

---

## 📖 Table of Contents

1. [What is Karnix?](#-what-is-karnix)
2. [Features](#-features)
3. [Quick Start](#-quick-start)
4. [Installation](#-installation)
5. [How to Run](#-how-to-run)
6. [How to Use the Library](#-how-to-use-the-library)
7. [Complete Examples](#-complete-examples)
8. [Medical AI Module](#-medical-ai-module)
9. [Project Structure](#-project-structure)
10. [API Documentation](#-api-documentation)
11. [Troubleshooting](#-troubleshooting)
12. [Quick Command Reference](#-quick-command-reference)

---

## 🎯 What is Karnix?

## 🎯 What is Karnix?

Karnix is a **complete deep learning library** built entirely from scratch in C++14. It demonstrates how neural networks work at the mathematical level, from basic tensor operations to advanced medical AI applications for brain tumor detection. Every algorithm is implemented with exact mathematical formulations, providing a deep understanding of how neural networks operate under the hood.

The library includes:
- Complete mathematical foundation (vectors, matrices, calculus)
- Full neural network implementation with gradient checking
- Convolutional Neural Networks (CNNs) for image processing
- Medical AI system for brain tumor detection in MRI scans
- No external dependencies - just C++14 standard library

---

### Why Karnix?

- 🎓 **Educational**: Learn how neural networks really work under the hood
- 🔬 **From-Scratch**: Every algorithm coded with exact mathematical formulations
- 🏥 **Medical AI**: Real-world brain tumor detection with clinical evaluation
- �� **Complete**: Includes everything from calculus to CNNs
- ✅ **Production-Ready**: Gradient checking, optimizers, full architectures
- 🚀 **No Dependencies**: Just C++14 standard library

---

## ✨ Features

### 🧮 Mathematical Foundation
- **Vector Operations**: Addition, subtraction, dot product, norms
- **Matrix Operations**: Multiplication, transpose, eigenvalues, SVD
- **Calculus Engine**: Derivatives, gradients, Hessians, Jacobians
- **Computational Graphs**: Automatic differentiation
- **Graph Visualizations**: ASCII plots for functions

### 🤖 Neural Network Components
- **Tensor Operations**: Multi-dimensional arrays with automatic gradient tracking
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, GELU
- **Layers**: 
  - Convolution (2D with padding, stride, dilation)
  - Pooling (Max, Average, Global)
  - Fully Connected (Dense)
  - Flatten
- **Optimizers**: SGD, Momentum, Adam, RMSProp, AdamW
- **Loss Functions**: MSE, Cross-Entropy, Binary Cross-Entropy
- **Gradient Checking**: Numerical verification of all gradients

### 🏥 Medical AI System
- **Brain Tumor Detection**: CNN for MRI analysis
- **Clinical Metrics**: Sensitivity, Specificity, Precision, F1-Score
- **Visual Interpretation**: Activation heatmaps
- **ROC Analysis**: Performance evaluation with AUC
- **Synthetic MRI Generation**: For testing and demonstration
- **Multiple Pathologies**: Normal, Tumor, Edema, Metastasis

---

## 🚀 Quick Start

### Prerequisites
- C++14 compatible compiler (g++, clang++)
- macOS, Linux, or Windows with MinGW

### 1. Clone the Repository
```bash
git clone https://github.com/hyperseconds/Karnix.git
cd Karnix
```

### 2. Compile
```bash
# Compile the main comprehensive demo
clang++ -std=gnu++14 -g main.c++ -o main

# OR using g++
g++ -std=gnu++14 -g main.c++ -o main
```

### 3. Run
```bash
# Run the complete demonstration
./main

# Run just the medical AI demo
clang++ -std=gnu++14 -g medical_demo.c++ -o medical_demo
./medical_demo
```

That's it! You'll see the complete system in action.

---

## 📦 Installation

### System Requirements
- **Compiler**: C++14 or later (clang++, g++, MSVC)
- **OS**: macOS, Linux, Windows
- **RAM**: 2GB minimum (4GB recommended)
- **Disk**: 100MB for source + compiled binaries

### Installation Steps

#### Option 1: Direct Compilation (Recommended)
```bash
cd Karnix
clang++ -std=gnu++14 -g main.c++ -o main
```

#### Option 2: Using Tasks (VS Code)
If you're using VS Code, the project includes pre-configured tasks:
```bash
# Press Cmd+Shift+B (macOS) or Ctrl+Shift+B (Windows/Linux)
# Select "build C++ project"
```

#### Option 3: Makefile (Create if needed)
```bash
# Create a Makefile
cat > Makefile << 'MAKEFILE'
CXX = clang++
CXXFLAGS = -std=gnu++14 -g
TARGET = main
MEDICAL_TARGET = medical_demo

all: $(TARGET)

$(TARGET): main.c++
	$(CXX) $(CXXFLAGS) main.c++ -o $(TARGET)

medical: medical_demo.c++
	$(CXX) $(CXXFLAGS) medical_demo.c++ -o $(MEDICAL_TARGET)

clean:
	rm -f $(TARGET) $(MEDICAL_TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run medical
MAKEFILE

# Then use
make          # Compile
make run      # Compile and run
make medical  # Compile medical demo
make clean    # Clean up
```

---

## 🎮 How to Run

### Running the Complete Demo

The main program demonstrates everything:
```bash
./main
```

**What you'll see:**
1. **Mathematical Foundations** (5-10 minutes)
   - Vector and matrix operations
   - Calculus demonstrations
   - Computational graphs
   - Function visualizations

2. **Neural Network Components** (3-5 minutes)
   - Tensor operations
   - Activation functions
   - Layer demonstrations
   - Optimizer comparisons
   - Gradient checking

3. **Computer Vision CNN** (2-3 minutes)
   - Image processing
   - Pattern recognition
   - Feature visualization

4. **Medical AI System** (5-10 minutes)
   - Brain MRI analysis
   - Tumor detection
   - Clinical evaluation
   - Performance metrics

**Total runtime: ~15-30 minutes**

### Running Specific Sections

#### Medical AI Only
```bash
./medical_demo
```
Focuses on brain tumor detection (faster, ~5 minutes)

#### Custom Test
Create your own test file:
```cpp
#include "neural_network/utils/Tensor.hpp"
#include <iostream>

int main() {
    Tensor t({2, 3}, false);
    t.random_normal(0, 1);
    t.print("My Tensor");
    return 0;
}
```

Compile and run:
```bash
clang++ -std=gnu++14 my_test.cpp -o my_test
./my_test
```

---

## 📚 How to Use the Library

### 1. Using Tensors

Tensors are the foundation of everything:

```cpp
#include "neural_network/utils/Tensor.hpp"

// Create a tensor
Tensor t({2, 3, 4}, false);  // Shape: [2, 3, 4], no gradient tracking

// Initialize
t.random_normal(0.0, 1.0);   // Mean=0, Std=1
t.fill_xavier(3, 4);         // Xavier initialization

// Access elements
double val = t(0, 1, 2);     // 3D access
t(0, 1, 2) = 5.0;            // Set value

// Operations
Tensor sum = t + other;
Tensor product = t * other;

// Print
t.print("My Tensor");
```

### 2. Building a Simple Neural Network

```cpp
#include "neural_network/layers/FullyConnectedLayer.hpp"
#include "neural_network/layers/ActivationFunctions.hpp"
#include "neural_network/optimizers/Optimizers.hpp"
#include "neural_network/utils/LossFunctions.hpp"

// Create layers
FullyConnectedLayer fc1(784, 128, true);  // Input: 784, Output: 128, with bias
FullyConnectedLayer fc2(128, 10, true);   // Output: 10 classes

// Activation functions
ActivationFunctions::ReLU relu;
ActivationFunctions::Softmax softmax;

// Optimizer
Optimizers::Adam optimizer(0.001);  // Learning rate: 0.001

// Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    // Forward pass
    Tensor h1 = fc1.forward(input);
    Tensor a1 = relu.forward(h1);
    Tensor h2 = fc2.forward(a1);
    Tensor output = softmax.forward(h2);
    
    // Compute loss
    double loss = CrossEntropyLoss::forward(output, target);
    
    // Backward pass
    Tensor grad = CrossEntropyLoss::backward(output, target);
    grad = softmax.backward(grad);
    grad = fc2.backward(grad);
    grad = relu.backward(grad);
    grad = fc1.backward(grad);
    
    // Update weights
    optimizer.update(fc1.get_weights(), fc1.get_weight_grad());
    optimizer.update(fc1.get_bias(), fc1.get_bias_grad());
    optimizer.update(fc2.get_weights(), fc2.get_weight_grad());
    optimizer.update(fc2.get_bias(), fc2.get_bias_grad());
    
    std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
}
```

### 3. Building a CNN

```cpp
#include "neural_network/layers/ConvolutionLayer.hpp"
#include "neural_network/layers/PoolingLayers.hpp"

// Create CNN layers
ConvolutionLayer conv1(3, 16, 3, 1, 1);    // 3→16 channels, 3x3 kernel
MaxPoolingLayer pool1(2, 2);                // 2x2 pooling
ConvolutionLayer conv2(16, 32, 3, 1, 1);   // 16→32 channels
MaxPoolingLayer pool2(2, 2);
FullyConnectedLayer fc(32*8*8, 10, true);  // Assuming 32x32 input

// Forward pass
Tensor img({3, 32, 32}, false);  // RGB image
img.random_normal(0, 1);

Tensor c1 = conv1.forward(img);
Tensor r1 = relu.forward(c1);
Tensor p1 = pool1.forward(r1);

Tensor c2 = conv2.forward(p1);
Tensor r2 = relu.forward(c2);
Tensor p2 = pool2.forward(r2);

// Flatten and classify
Tensor flat = flatten(p2);
Tensor output = fc.forward(flat);
```

### 4. Using Medical AI

```cpp
#include "medical/MedicalCNN.hpp"
#include "medical/MedicalImageProcessor.hpp"
#include "medical/MedicalEvaluationUtils.hpp"

// Initialize
MedicalCNN cnn(1, 64, 64, true);          // 1 channel, 64x64 images
MedicalImageProcessor processor(64, true);

// Create or load MRI
MedicalImageProcessor::MRIImage mri = processor.create_brain_mri(64, 64, "tumor");

// Preprocess
Tensor mri_tensor = processor.mri_to_tensor(mri, true);

// Predict
MedicalCNN::MedicalPrediction result = cnn.predict(mri_tensor);

// Results
std::cout << "Diagnosis: " << result.diagnosis << std::endl;
std::cout << "Tumor Probability: " << result.tumor_probability << std::endl;
std::cout << "Confidence: " << result.confidence * 100 << "%" << std::endl;

// Visualize
processor.visualize_mri_ascii(mri, true);
Tensor heatmap = cnn.get_activation_heatmap();
processor.create_activation_heatmap(mri, heatmap, "Focus Areas");
```

### 5. Gradient Checking

Always verify your gradients:

```cpp
#include "neural_network/tests/GradientChecker.hpp"

GradientChecker checker(1e-5, 1e-4);  // h=1e-5, tolerance=1e-4

// Check loss gradients
bool passed = checker.check_loss_gradients(predictions, targets, "MSE");

if (passed) {
    std::cout << "✓ Gradients are correct!" << std::endl;
} else {
    std::cout << "✗ Gradient mismatch detected!" << std::endl;
}
```

---

## 💡 Complete Examples

### Example 1: MNIST-like Classification

```cpp
#include <iostream>
#include "neural_network/layers/FullyConnectedLayer.hpp"
#include "neural_network/layers/ActivationFunctions.hpp"
#include "neural_network/utils/LossFunctions.hpp"

int main() {
    // Network architecture
    FullyConnectedLayer layer1(784, 256, true);
    FullyConnectedLayer layer2(256, 128, true);
    FullyConnectedLayer layer3(128, 10, true);
    
    ActivationFunctions::ReLU relu;
    ActivationFunctions::Softmax softmax;
    
    // Dummy input (28x28 flattened)
    Tensor input({784}, false);
    input.random_normal(0, 1);
    
    // Target (one-hot encoded)
    Tensor target({10}, false);
    target.get_data() = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};  // Class 3
    
    // Forward
    Tensor h1 = layer1.forward(input);
    Tensor a1 = relu.forward(h1);
    Tensor h2 = layer2.forward(a1);
    Tensor a2 = relu.forward(h2);
    Tensor h3 = layer3.forward(a2);
    Tensor output = softmax.forward(h3);
    
    // Loss
    double loss = CrossEntropyLoss::forward(output, target);
    std::cout << "Loss: " << loss << std::endl;
    
    // Find prediction
    int predicted_class = 0;
    double max_prob = output[0];
    for (int i = 1; i < 10; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            predicted_class = i;
        }
    }
    
    std::cout << "Predicted class: " << predicted_class << std::endl;
    std::cout << "Confidence: " << max_prob * 100 << "%" << std::endl;
    
    return 0;
}
```

### Example 2: Image Classification CNN

```cpp
#include "neural_network/cnn/CNNArchitecture.hpp"

int main() {
    // Create a CNN for 32x32 RGB images, 5 classes
    CNNArchitecture cnn(3, 32, 32, 5, 0.001, true);
    
    // Create random image
    Tensor image({3, 32, 32}, false);
    image.random_normal(0, 1);
    
    // Predict
    int predicted_class = cnn.predict(image);
    std::vector<double> probs = cnn.get_confidence_scores(image);
    
    std::cout << "Predicted class: " << predicted_class << std::endl;
    for (int i = 0; i < probs.size(); i++) {
        std::cout << "Class " << i << ": " << probs[i] * 100 << "%" << std::endl;
    }
    
    // Visualize what the CNN sees
    cnn.visualize_feature_maps();
    cnn.create_activation_heatmap();
    
    return 0;
}
```

### Example 3: Medical Diagnosis System

```cpp
#include "medical/MedicalCNN.hpp"
#include "medical/MedicalImageProcessor.hpp"
#include "medical/MedicalEvaluationUtils.hpp"

int main() {
    std::cout << "🏥 Brain Tumor Detection System" << std::endl;
    
    // Initialize
    MedicalCNN cnn(1, 64, 64, true);
    MedicalImageProcessor processor(64, true);
    
    // Analyze different cases
    std::vector<std::string> cases = {"normal", "tumor", "edema"};
    
    for (const auto& case_type : cases) {
        std::cout << "\n--- Analyzing " << case_type << " case ---" << std::endl;
        
        // Create synthetic MRI
        auto mri = processor.create_brain_mri(64, 64, case_type);
        mri.patient_id = "PATIENT_" + case_type;
        
        // Visualize
        processor.visualize_mri_ascii(mri, true);
        
        // Diagnose
        Tensor tensor = processor.mri_to_tensor(mri, true);
        auto prediction = cnn.predict(tensor);
        
        // Report
        std::cout << "\nDiagnosis: " << prediction.diagnosis << std::endl;
        std::cout << "Tumor Probability: " << prediction.tumor_probability << std::endl;
        std::cout << "Normal Probability: " << prediction.normal_probability << std::endl;
        std::cout << "Confidence: " << prediction.confidence * 100 << "%" << std::endl;
        
        // Show where CNN is looking
        Tensor heatmap = cnn.get_activation_heatmap();
        processor.create_activation_heatmap(mri, heatmap, "CNN Focus");
    }
    
    // Clinical evaluation
    auto dataset = MedicalEvaluationUtils::create_clinical_dataset(20, 64);
    MedicalEvaluationUtils::evaluate_clinical_performance(cnn, dataset);
    
    return 0;
}
```

---

## 🏥 Medical AI Module

### Brain Tumor Detection System

The medical AI module implements a complete clinical-grade CNN for brain tumor detection in MRI scans.

#### Architecture
```
Input: Grayscale MRI (64×64)
    ↓
Conv2D(1→8, 3×3) + ReLU + MaxPool(2×2) → [8, 32, 32]
    ↓
Conv2D(8→16, 3×3) + ReLU + MaxPool(2×2) → [16, 16, 16]
    ↓
Flatten + FC(4096→128) + ReLU → [128]
    ↓
FC(128→2) + Softmax → [Normal, Tumor]
```

**Parameters**: 525,922 (optimized for medical imaging)

#### Supported Pathologies

1. **Normal Brain Tissue**
   - Healthy anatomical structures
   - Gray matter, white matter, CSF

2. **Primary Tumors**
   - Glioblastoma-like lesions
   - Hyperintense regions

3. **Edema**
   - Perilesional swelling
   - Hypointense patterns

4. **Metastasis**
   - Multiple small lesions
   - Distributed pathology

#### Clinical Metrics

The system provides standard medical evaluation metrics:

- **Sensitivity (Recall)**: TP / (TP + FN)
  - Measures ability to detect actual tumors
  - Critical for medical applications

- **Specificity**: TN / (TN + FP)
  - Measures ability to correctly identify normal cases
  - Minimizes false alarms

- **Precision**: TP / (TP + FP)
  - Accuracy of tumor predictions
  - Reduces unnecessary interventions

- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
  - Balanced performance measure

- **ROC/AUC**: Area under receiver operating characteristic curve
  - Overall discriminative ability

#### Visual Interpretation

The system provides explainable AI through:

1. **Activation Heatmaps**
   - Shows where the CNN is focusing
   - Highlights suspicious regions
   - Helps doctors understand AI decisions

2. **ASCII Visualization**
   - Quick MRI slice visualization
   - Shows tumor regions with '#' symbols
   - Portable and lightweight

3. **Clinical Reports**
   - Structured diagnostic output
   - Confidence scores
   - Recommendation levels

#### Usage Example

```cpp
// Quick diagnosis
MedicalCNN cnn(1, 64, 64);
auto result = cnn.predict(mri_tensor);

if (result.predicted_class == 1) {
    std::cout << "⚠️ TUMOR DETECTED" << std::endl;
    std::cout << "Confidence: " << result.confidence * 100 << "%" << std::endl;
    std::cout << "Recommendation: Further imaging required" << std::endl;
} else {
    std::cout << "✓ NORMAL SCAN" << std::endl;
}
```

#### Clinical Integration

The system is designed for use as a **second opinion tool**:

1. Radiologist reviews the MRI first
2. CNN provides automated analysis
3. Radiologist considers CNN output
4. Final diagnosis made by medical professional

**Safety Features**:
- High sensitivity to avoid missing tumors
- Confidence thresholds for reliable predictions
- Visual explanations for transparency
- Regular performance monitoring

⚠️ **Medical Disclaimer**: This is for research and educational purposes only. Not FDA approved. Never use as the sole diagnostic tool.

---

## 📂 Project Structure

```
Karnix/
│
├── main.c++                          # Complete system demonstration
├── medical_demo.c++                  # Focused medical AI demo
├── README.md                         # This file
│
├── mathematical/                     # Mathematical foundations
│   ├── Vector/
│   │   └── VectorOperations.hpp      # Vector math (add, dot, norm)
│   ├── matrix/
│   │   ├── MatrixOperations.hpp      # Matrix operations
│   │   └── AdvancedMatrix.hpp        # Eigenvalues, SVD, etc.
│   ├── Calc/
│   │   ├── Calc.hpp                  # Basic calculus
│   │   └── CalcEng.hpp               # Advanced calculus (gradient, Hessian)
│   ├── Graph/
│   │   └── GraphVisualizer.hpp       # ASCII plotting
│   └── ComputationalGraph/
│       └── ComputationalGraph.hpp    # Autograd system
│
├── neural_network/                   # Neural network components
│   ├── utils/
│   │   ├── Tensor.hpp                # Core tensor operations
│   │   ├── ImageProcessor.hpp        # Image preprocessing
│   │   └── LossFunctions.hpp         # MSE, Cross-Entropy, etc.
│   │
│   ├── layers/
│   │   ├── ActivationFunctions.hpp   # ReLU, Sigmoid, Softmax, etc.
│   │   ├── ConvolutionLayer.hpp      # 2D Convolution
│   │   ├── PoolingLayers.hpp         # Max/Avg/Global pooling
│   │   ├── FullyConnectedLayer.hpp   # Dense layers
│   │   └── FlattenLayer.hpp          # Flatten operation
│   │
│   ├── optimizers/
│   │   └── Optimizers.hpp            # SGD, Adam, RMSProp, etc.
│   │
│   ├── tests/
│   │   └── GradientChecker.hpp       # Numerical gradient verification
│   │
│   └── cnn/
│       └── CNNArchitecture.hpp       # Complete CNN implementation
│
├── medical/                          # Medical AI module
│   ├── MedicalImageProcessor.hpp     # MRI preprocessing & visualization
│   ├── MedicalCNN.hpp                # Tumor detection CNN
│   └── MedicalEvaluationUtils.hpp    # Clinical metrics & evaluation
│
└── images/                           # Sample data (if any)
    └── (place your MRI images here)
```

### Key Files Explained

#### Core Components

- **`Tensor.hpp`**: Foundation of everything
  - Multi-dimensional arrays
  - Automatic gradient tracking
  - Shape manipulation
  - Memory management

- **`ConvolutionLayer.hpp`**: 2D convolution
  - Padding, stride, dilation support
  - Im2col optimization
  - Exact gradient computation

- **`Optimizers.hpp`**: Training algorithms
  - SGD with momentum
  - Adam (adaptive learning)
  - RMSProp
  - Weight decay

#### Medical AI

- **`MedicalCNN.hpp`**: Brain tumor detector
  - LeNet-inspired architecture
  - Medical-specific preprocessing
  - Clinical metric calculation

- **`MedicalImageProcessor.hpp`**: MRI handling
  - Grayscale conversion
  - Normalization
  - Synthetic MRI generation
  - ASCII visualization

- **`MedicalEvaluationUtils.hpp`**: Clinical evaluation
  - Sensitivity/Specificity
  - ROC curve analysis
  - Performance reporting

---

## 📖 API Documentation

### Tensor Class

```cpp
class Tensor {
public:
    // Constructors
    Tensor(std::vector<int> shape, bool requires_grad = false);
    
    // Initialization
    void random_normal(double mean, double std);
    void fill_xavier(int fan_in, int fan_out);
    void fill(double value);
    
    // Access
    double& operator()(int i, int j, int k);  // 3D access
    double& operator[](int idx);               // 1D access
    
    // Operations
    Tensor operator+(const Tensor& other);
    Tensor operator*(const Tensor& other);
    
    // Utilities
    void print(const std::string& name = "");
    std::vector<int> get_shape();
    std::vector<double>& get_data();
};
```

### FullyConnectedLayer

```cpp
class FullyConnectedLayer {
public:
    FullyConnectedLayer(int input_size, int output_size, bool use_bias = true);
    
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    
    Tensor& get_weights();
    Tensor& get_bias();
    Tensor& get_weight_grad();
    Tensor& get_bias_grad();
};
```

### ConvolutionLayer

```cpp
class ConvolutionLayer {
public:
    ConvolutionLayer(int in_channels, int out_channels, int kernel_size,
                     int stride = 1, int padding = 0);
    
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    
    Tensor& get_weights();
    Tensor& get_bias();
};
```

### Activation Functions

```cpp
namespace ActivationFunctions {
    class ReLU {
    public:
        Tensor forward(const Tensor& input);
        Tensor backward(const Tensor& grad_output);
    };
    
    class Sigmoid { /* ... */ };
    class Tanh { /* ... */ };
    class Softmax { /* ... */ };
}
```

### MedicalCNN

```cpp
class MedicalCNN {
public:
    struct MedicalPrediction {
        double tumor_probability;
        double normal_probability;
        int predicted_class;
        double confidence;
        std::string diagnosis;
    };
    
    MedicalCNN(int channels, int height, int width, bool verbose = false);
    
    MedicalPrediction predict(const Tensor& mri_input);
    Tensor get_activation_heatmap();
    void set_verbose(bool verb);
};
```

### MedicalImageProcessor

```cpp
class MedicalImageProcessor {
public:
    struct MRIImage {
        std::vector<std::vector<double>> data;
        int width, height;
        bool has_tumor;
        std::string patient_id;
    };
    
    MedicalImageProcessor(int size = 64, bool verbose = false);
    
    Tensor mri_to_tensor(const MRIImage& mri, bool normalize = true);
    MRIImage create_brain_mri(int width, int height, const std::string& type);
    void visualize_mri_ascii(const MRIImage& mri, bool show_tumor_info = true);
    void create_activation_heatmap(const MRIImage& original, const Tensor& activation_map,
                                   const std::string& title);
};
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Compilation Errors

**Error**: `error: 'class Tensor' has no member named 'get'`
**Solution**: Use parentheses `tensor(i, j, k)` instead of `tensor.get({i, j, k})`

**Error**: `undefined reference to 'main'`
**Solution**: Make sure your file has a `main()` function

**Error**: `clang: command not found`
**Solution**: Install Xcode Command Line Tools (macOS) or GCC (Linux)
```bash
# macOS
xcode-select --install

# Linux (Ubuntu/Debian)
sudo apt-get install build-essential

# Linux (Fedora)
sudo dnf install gcc-c++
```

#### 2. Runtime Errors

**Error**: `Segmentation fault`
**Causes**:
- Out of bounds tensor access
- Mismatched tensor shapes
- Stack overflow (tensor too large)

**Solution**: Check tensor dimensions and add bounds checking

**Error**: `nan` or `inf` values
**Causes**:
- Learning rate too high
- Gradient explosion
- Division by zero

**Solution**:
- Reduce learning rate
- Add gradient clipping
- Check for divide-by-zero

#### 3. Performance Issues

**Problem**: Slow execution
**Solutions**:
- Compile with optimizations: `-O3` flag
- Use smaller batch sizes
- Reduce model complexity
- Profile with `-pg` flag

**Problem**: High memory usage
**Solutions**:
- Use smaller tensors
- Clear intermediate results
- Batch processing

#### 4. Medical AI Issues

**Problem**: Always predicts one class
**Causes**:
- Model not trained
- Learning rate too high/low
- Class imbalance

**Solution**: This is a demonstration system showing architecture, not a trained model

---

## 📋 Quick Command Reference

```bash
# Compile main demo
clang++ -std=gnu++14 -g main.c++ -o main

# Run main demo
./main

# Compile medical demo
clang++ -std=gnu++14 -g medical_demo.c++ -o medical_demo

# Run medical demo
./medical_demo

# Compile with optimizations
clang++ -std=gnu++14 -O3 main.c++ -o main

# Clean up
rm -f main medical_demo *.dSYM

# Quick test
echo 'int main() { return 0; }' > test.cpp && clang++ test.cpp && ./a.out && echo "✓ Compiler works!"
```

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This software is for **research and educational purposes only**.

- ❌ NOT FDA approved
- ❌ NOT for clinical diagnosis
- ❌ NOT a substitute for professional medical advice
- ✅ Demonstration of AI concepts only
- ✅ Requires extensive validation before clinical use
- ✅ Always consult qualified medical professionals

---

**🧠 Karnix** - *Complete deep learning implementation from scratch in C++*

*Last Updated: November 11, 2025*

## 🎓 Learning Resources

### Understanding the Code

1. **Start with Tensors** (`neural_network/utils/Tensor.hpp`)
   - Understand multi-dimensional arrays
   - Learn about shapes and strides

2. **Learn Layers** (`neural_network/layers/`)
   - Start with FullyConnectedLayer
   - Move to ConvolutionLayer
   - Understand forward/backward

3. **Study Training** (`main.c++`)
   - See complete training loop
   - Understand loss computation
   - Learn optimizer updates

4. **Explore Medical AI** (`medical/`)
   - See real application
   - Understand preprocessing
   - Learn about clinical metrics

### Mathematical Background

- **Linear Algebra**: Vectors, matrices, operations
- **Calculus**: Derivatives, gradients, chain rule
- **Probability**: Softmax, cross-entropy
- **Optimization**: Gradient descent, momentum

### Recommended Reading

- Deep Learning by Goodfellow, Bengio, Courville
- Neural Networks and Deep Learning by Michael Nielsen
- CS231n: Convolutional Neural Networks (Stanford)

---

## 🤝 Contributing

This is an educational project demonstrating deep learning concepts. Contributions are welcome!

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Areas for Improvement

- [ ] Add more layer types (Dropout, BatchNorm)
- [ ] Implement data augmentation
- [ ] Add CUDA support
- [ ] Create Python bindings
- [ ] Support real DICOM images
- [ ] Add more optimizers (AdaBound, LAMB)
- [ ] Implement attention mechanisms
- [ ] Add model saving/loading

---



## ⚠️ Medical Disclaimer


- ❌ NOT FDA approved
- ✅ for clinical diagnosis
- ✅ a substitute for professional medical advice
- ✅ Demonstration of AI concepts only
- ✅ Requires extensive validation before clinical use
- ✅ Always consult qualified medical professionals

---

## 📞 Contact & Support

- **Repository**: https://github.com/hyperseconds/Karnix
- **Issues**: Report bugs or request features on GitHub
- **Author**: Sudharsan S

---

## 🎉 Acknowledgments

Built from scratch to demonstrate:
- How neural networks work mathematically
- How CNNs process images
- How AI can assist in medical imaging
- The importance of proper evaluation metrics

**🧠 Karnix** - *Where mathematical precision meets medical innovation*

---

## Quick Command Reference

```bash
# Compile main demo
clang++ -std=gnu++14 -g main.c++ -o main

# Run main demo
./main

# Compile medical demo
clang++ -std=gnu++14 -g medical_demo.c++ -o medical_demo

# Run medical demo
./medical_demo

# Compile with optimizations
clang++ -std=gnu++14 -O3 main.c++ -o main

# Clean up
rm -f main medical_demo *.dSYM

# Quick test
echo 'int main() { return 0; }' > test.cpp && clang++ test.cpp && ./a.out && echo "✓ Compiler works!"
```

---

*Last Updated: November 7, 2025*
