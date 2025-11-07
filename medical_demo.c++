#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

#include "medical/MedicalImageProcessor.hpp"
#include "medical/MedicalCNN.hpp"
#include "medical/MedicalEvaluationUtils.hpp"

using namespace std;

int main() {
    cout << "🏥 MEDICAL AI DEMONSTRATION: Brain Tumor Detection" << endl;
    cout << string(60, '=') << endl;
    
    // Initialize medical CNN
    cout << "\n1. Initializing Medical CNN Architecture..." << endl;
    MedicalCNN medical_cnn(1, 64, 64, true);
    
    // Create synthetic clinical cases
    cout << "\n2. Creating Synthetic Clinical Cases..." << endl;
    MedicalImageProcessor processor(64, true);
    
    vector<string> case_types = {"normal", "tumor", "edema", "metastasis"};
    
    for (const string& case_type : case_types) {
        cout << "\n--- Analyzing " << case_type << " case ---" << endl;
        
        // Create MRI case
        MedicalImageProcessor::MRIImage mri_case = processor.create_brain_mri(64, 64, case_type);
        mri_case.patient_id = "DEMO_" + case_type;
        
        // Visualize MRI
        processor.visualize_mri_ascii(mri_case, true);
        
        // Convert to tensor and analyze
        Tensor mri_tensor = processor.mri_to_tensor(mri_case, true);
        
        // Get medical prediction
        medical_cnn.set_verbose(false);  // Reduce output for demo
        MedicalCNN::MedicalPrediction prediction = medical_cnn.predict(mri_tensor);
        
        cout << "\nClinical Assessment:" << endl;
        cout << "- Medical Diagnosis: " << prediction.diagnosis << endl;
        cout << "- Confidence Level: " << fixed << setprecision(1) << (prediction.confidence * 100) << "%" << endl;
        cout << "- Normal Probability: " << setprecision(3) << prediction.normal_probability << endl;
        cout << "- Tumor Probability: " << setprecision(3) << prediction.tumor_probability << endl;
        
        // Show activation heatmap
        Tensor heatmap = medical_cnn.get_activation_heatmap();
        processor.create_activation_heatmap(mri_case, heatmap, "CNN Focus Areas");
    }
    
    // Quick clinical evaluation
    cout << "\n3. Clinical Performance Evaluation..." << endl;
    MedicalEvaluationUtils::ClinicalDataset clinical_data = 
        MedicalEvaluationUtils::create_clinical_dataset(12, 64);
    
    MedicalEvaluationUtils::evaluate_clinical_performance(medical_cnn, clinical_data);
    
    cout << "\n" << string(60, '=') << endl;
    cout << "Medical AI Demonstration Complete!" << endl;
    cout << "✓ Brain tumor detection CNN implemented" << endl;
    cout << "✓ Clinical evaluation metrics calculated" << endl;
    cout << "✓ Visual interpretation provided" << endl;
    cout << "✓ Ready for further development and validation" << endl;
    cout << string(60, '=') << endl;
    
    return 0;
}