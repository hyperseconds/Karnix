#include <iostream>
#include <vector>
#include "mathematical/Vector/VectorOperations.hpp"
#include "mathematical/matrix/MatrixOperations.hpp"
#include "mathematical/matrix/AdvancedMatrix.hpp"

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

    return 0;
}