#include <iostream>
#include <vector>
#include "mathematical/Vector/VectorOperations.hpp"
#include "mathematical/matrix/MatrixOperations.hpp"
#include "mathematical/matrix/AdvancedMatrix.hpp"
#include "mathematical/Calc/Calc.hpp"
#include "mathematical/Calc/CalcEng.hpp"
#include "mathematical/Graph/GraphVisualizer.hpp"

using namespace std;

int main(){
    cout << "=== Vector Operations ===" << endl;
    VectorOperations vo;
    int arr1[] = {3, 4, 5};
    int arr2[] = {1, 2, 3};
    vo.setVectors(arr1, 3, arr2, 3);
    vo.displayVector1();
    vo.displayVector2();
    
    vo.vector_add();
    vo.displayAddResult();
    
    vo.vector_sub();
    vo.displaySubResult();
    
    vo.scalarVectorMultiply(2);
    vo.displayScalarResult();
    
    vo.vector_norm();
    vo.displayNormResult();

    cout << "\n=== Matrix Operations ===" << endl;
    MatrixOperations mo;
    
    vector<vector<int>> matrix1 = {
        {1, 2, 3},
        {4, 5, 6}
    };
    
    vector<vector<int>> matrix2 = {
        {7, 8, 9},
        {10, 11, 12}
    };
    
    mo.setMatrices(matrix1, matrix2);
    mo.displayMatrix1();
    mo.displayMatrix2();
    
    mo.matrix_add();
    mo.displayAddResult();
    
    mo.matrix_sub();
    mo.displaySubResult();
    
    mo.setMatrix(matrix1);
    mo.scalar_multiply(3);
    mo.displayScalarResult();
    
    mo.matrix_transpose();
    mo.displayTransposeResult();
    
    mo.displayNorm();

    cout << "\n=== Advanced Matrix Operations ===" << endl;
    AdvancedMatrix am;
    
    // Create a 2x2 matrix for eigenvalue demonstration
    vector<vector<double>> square_matrix = {
        {4.0, 2.0},
        {1.0, 3.0}
    };
    
    am.setMatrix(square_matrix);
    am.displayMatrix();
    
    // Matrix trace
    am.displayTrace();
    
    // Matrix rank
    am.displayRank();
    
    // Eigenvalues (conceptual)
    am.displayEigenvalues();
    
    // Eigenvectors (conceptual)
    am.displayEigenvectors();
    
    // Matrix projection
    vector<double> vector_b = {5.0, 3.0};
    am.displayProjection(vector_b);
    
    // Test with a 3x3 matrix
    cout << "\n=== 3x3 Matrix Example ===" << endl;
    vector<vector<double>> matrix_3x3 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    am.setMatrix(matrix_3x3);
    am.displayMatrix();
    am.displayTrace();
    am.displayRank();
    am.displayEigenvalues(); // Will show conceptual message
    
    vector<double> vector_b2 = {1.0, 2.0, 3.0};
    am.displayProjection(vector_b2);

    cout << "\n=== Calculus Operations ===" << endl;
    Calc calc;
    double x = 2.0;
    double y = 3.0;
    
    // Constant function
    cout << "Constant function f(x) = 5: " << calc.constant(x, 5.0) << endl;
    cout << "Derivative of constant: " << calc.constant_derivative(x) << endl;
    
    // Linear function
    cout << "Linear function f(x) = 2x + 3: " << calc.linear(x, 2.0, 3.0) << endl;
    
    // Polynomial function
    vector<double> coeffs = {1.0, 2.0, 3.0}; // 1 + 2x + 3x²
    cout << "Polynomial f(x) = 1 + 2x + 3x² at x=" << x << ": " << calc.polynomial(x, coeffs) << endl;
    cout << "Polynomial derivative at x=" << x << ": " << calc.polynomial_derivative(x, coeffs) << endl;
    
    // Exponential function
    cout << "Exponential e^x at x=" << x << ": " << calc.exponential(x) << endl;
    cout << "Exponential derivative at x=" << x << ": " << calc.exponential_derivative(x) << endl;
    
    // Logarithmic function
    cout << "Natural log ln(x) at x=" << x << ": " << calc.logarithmic(x) << endl;
    cout << "Logarithmic derivative at x=" << x << ": " << calc.logarithmic_derivative(x) << endl;
    
    // Trigonometric functions
    cout << "Sine sin(x) at x=" << x << ": " << calc.sine(x) << endl;
    cout << "Sine derivative cos(x) at x=" << x << ": " << calc.sine_derivative(x) << endl;
    cout << "Cosine cos(x) at x=" << x << ": " << calc.cosine(x) << endl;
    cout << "Cosine derivative -sin(x) at x=" << x << ": " << calc.cosine_derivative(x) << endl;
    
    // Composite function
    cout << "Composite function f(g(x)) where g(x)=x+1, f(u)=u² at x=" << x << ": " << calc.f_of_g(x) << endl;
    cout << "Composite derivative using chain rule at x=" << x << ": " << calc.composite_derivative(x) << endl;
    
    // Multivariate function
    cout << "Multivariate f(x,y) = x² + y² at (" << x << "," << y << "): " << calc.multivariate(x, y) << endl;
    auto gradient = calc.multivariate_gradient(x, y);
    cout << "Gradient at (" << x << "," << y << "): (" << gradient.first << ", " << gradient.second << ")" << endl;

    cout << "\n=== AI/ML Advanced Calculus ===" << endl;
    CalcEng calcEng;
    
    // Demonstrate the 5 AI calculus concepts
    calcEng.demonstrateAICalculus();
    
    // Individual demonstrations
    cout << "\n--- Individual AI Calculus Operations ---" << endl;
    
    // 1. Differentiation for neural network backpropagation
    auto relu = [&calcEng](double x) { return calcEng.neural_activation_relu(x); };
    calcEng.differentiation(relu, 2.0);
    calcEng.displayDifferentiation(2.0);
    
    // 2. Integration for probability calculations
    auto probability_density = [](double x) { return std::exp(-x*x/2) / std::sqrt(2*M_PI); };
    calcEng.integration(probability_density, -1.0, 1.0);
    calcEng.displayIntegration(-1.0, 1.0);
    
    // 3. Gradient for loss function optimization
    auto loss_func = [](const std::vector<double>& w) {
        return w[0]*w[0] + w[1]*w[1] + 2*w[0]*w[1]; // Simple quadratic loss
    };
    std::vector<double> weights = {1.5, 2.5};
    calcEng.gradient(loss_func, weights);
    calcEng.displayGradient(weights);
    
    // 4. Jacobian for neural network layer transformations
    std::vector<std::function<double(std::vector<double>)>> network_funcs = {
        [](const std::vector<double>& x) { return x[0] + x[1]; },
        [](const std::vector<double>& x) { return x[0] * x[1]; }
    };
    std::vector<double> inputs = {1.0, 2.0};
    calcEng.jacobian(network_funcs, inputs);
    calcEng.displayJacobian(inputs);
    
    // 5. Hessian for second-order optimization (Newton's method)
    auto optimization_func = [](const std::vector<double>& x) {
        return x[0]*x[0] + x[1]*x[1] - 2*x[0] - 4*x[1] + 5; // Convex quadratic
    };
    std::vector<double> opt_point = {0.5, 1.0};
    calcEng.hessian(optimization_func, opt_point);
    calcEng.displayHessian(opt_point);

    cout << "\n=== Graph Visualizations ===" << endl;
    GraphVisualizer graph;
    
    // 1. Sine Wave
    cout << "\n1. Trigonometric Function:" << endl;
    graph.plotSineWave();
    
    // 2. Quadratic Function
    cout << "\n2. Quadratic Function:" << endl;
    graph.plotQuadratic();
    
    // 3. Exponential Function
    cout << "\n3. Exponential Growth:" << endl;
    graph.plotExponential();
    
    // 4. Logarithmic Function
    cout << "\n4. Logarithmic Function:" << endl;
    graph.plotLogarithmic();
    
    // 5. Function and its Derivative
    cout << "\n5. Function and Derivative Comparison:" << endl;
    graph.plotDerivative();
    
    // 6. Gradient Field Visualization
    cout << "\n6. Gradient Field (AI Optimization Context):" << endl;
    graph.plotGradientField();
    
    // 7. Contour Plot
    cout << "\n7. Contour Plot (Loss Function Landscape):" << endl;
    graph.plotContour();
    
    // 8. Data Scatter Plot with Trend Line
    cout << "\n8. Data Analysis - Scatter Plot with Regression:" << endl;
    graph.plotDataScatter();
    
    // 9. Custom AI/ML Function Visualization
    cout << "\n9. Custom AI Function Visualization:" << endl;
    graph.setTitle("Neural Network Activation: Sigmoid vs ReLU");
    graph.setRange(-5, 5, -0.5, 5);
    graph.clearCanvas();
    graph.drawAxes();
    
    // Plot sigmoid activation function
    graph.plotFunction([](double x) { return 1.0 / (1.0 + std::exp(-x)); }, 's');
    
    // Plot ReLU activation function
    graph.plotFunction([](double x) { return std::max(0.0, x); }, 'r');
    
    graph.addScale();
    graph.display();
    cout << "Legend: s = Sigmoid, r = ReLU" << endl;
    
    // 10. Loss Function Visualization
    cout << "\n10. Loss Function Landscape:" << endl;
    graph.setTitle("Quadratic Loss Function: L = (x-2)² + (y-1)²");
    graph.setRange(-1, 5, -2, 4);
    graph.clearCanvas();
    graph.drawAxes();
    
    // Plot minimum point
    std::vector<std::pair<double, double>> minimum = {{2.0, 1.0}};
    graph.plotPoints(minimum, 'M');
    
    // Plot some gradient descent path
    std::vector<std::pair<double, double>> descent_path = {
        {4.5, 3.5}, {4.0, 3.0}, {3.5, 2.5}, {3.0, 2.0}, {2.5, 1.5}, {2.0, 1.0}
    };
    graph.plotPoints(descent_path, '*');
    
    graph.addScale();
    graph.display();
    cout << "M = Global Minimum, * = Gradient Descent Path" << endl;

    return 0;
}