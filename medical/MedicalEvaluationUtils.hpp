#pragma once
#include "MedicalCNN.hpp"
#include "MedicalImageProcessor.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>

class MedicalEvaluationUtils {
public:
    struct ClinicalDataset {
        std::vector<Tensor> mri_images;
        std::vector<int> labels;  // 0 = normal, 1 = tumor
        std::vector<std::string> patient_ids;
        std::vector<std::string> diagnoses;
    };
    
    struct ROCPoint {
        double threshold;
        double sensitivity;
        double specificity;
        double false_positive_rate;
    };

public:
    // Create synthetic clinical dataset for demonstration
    static ClinicalDataset create_clinical_dataset(int num_samples = 20, int image_size = 64) {
        ClinicalDataset dataset;
        MedicalImageProcessor processor(image_size, false);
        
        std::vector<std::string> tumor_types = {"tumor", "edema", "metastasis"};
        std::vector<std::string> normal_types = {"normal"};
        
        std::cout << "Creating clinical dataset..." << std::endl;
        std::cout << "Generating " << num_samples << " synthetic MRI cases" << std::endl;
        
        for (int i = 0; i < num_samples; ++i) {
            bool is_tumor = (i < num_samples / 2);  // Half tumor, half normal
            
            MedicalImageProcessor::MRIImage mri(image_size, image_size);
            std::string patient_id = "PATIENT_" + std::to_string(i + 1000);
            
            if (is_tumor) {
                std::string tumor_type = tumor_types[i % tumor_types.size()];
                mri = processor.create_brain_mri(image_size, image_size, tumor_type);
                mri.patient_id = patient_id;
                
                dataset.labels.push_back(1);
                dataset.diagnoses.push_back("Confirmed: " + tumor_type + " pathology");
            } else {
                mri = processor.create_brain_mri(image_size, image_size, "normal");
                mri.patient_id = patient_id;
                
                dataset.labels.push_back(0);
                dataset.diagnoses.push_back("Normal brain tissue");
            }
            
            // Convert to tensor
            Tensor mri_tensor = processor.mri_to_tensor(mri, true);
            dataset.mri_images.push_back(mri_tensor);
            dataset.patient_ids.push_back(patient_id);
            
            if ((i + 1) % 5 == 0) {
                std::cout << "Generated " << (i + 1) << "/" << num_samples << " cases" << std::endl;
            }
        }
        
        std::cout << "Dataset created successfully!" << std::endl;
        std::cout << "- Total cases: " << num_samples << std::endl;
        std::cout << "- Tumor cases: " << std::count(dataset.labels.begin(), dataset.labels.end(), 1) << std::endl;
        std::cout << "- Normal cases: " << std::count(dataset.labels.begin(), dataset.labels.end(), 0) << std::endl;
        
        return dataset;
    }
    
    // Perform comprehensive medical evaluation
    static void evaluate_clinical_performance(MedicalCNN& cnn, const ClinicalDataset& dataset) {
        std::cout << "\n=== CLINICAL EVALUATION IN PROGRESS ===" << std::endl;
        std::cout << "Testing CNN on " << dataset.mri_images.size() << " clinical cases..." << std::endl;
        
        // Collect all predictions
        std::vector<MedicalCNN::MedicalPrediction> predictions;
        std::vector<double> tumor_probabilities;
        
        for (size_t i = 0; i < dataset.mri_images.size(); ++i) {
            cnn.set_verbose(false);
            MedicalCNN::MedicalPrediction pred = cnn.predict(dataset.mri_images[i]);
            predictions.push_back(pred);
            tumor_probabilities.push_back(pred.tumor_probability);
        }
        
        // Calculate standard metrics
        MedicalCNN::TrainingMetrics metrics = cnn.evaluate_medical_performance(dataset.mri_images, dataset.labels);
        
        // Print detailed evaluation
        cnn.print_medical_evaluation(metrics);
        
        // Print case-by-case analysis
        print_case_by_case_analysis(dataset, predictions);
        
        // Generate ROC analysis
        generate_roc_analysis(dataset.labels, tumor_probabilities);
        
        // Clinical recommendations
        provide_clinical_recommendations(metrics);
    }
    
    // Generate ROC curve analysis
    static void generate_roc_analysis(const std::vector<int>& true_labels, 
                                    const std::vector<double>& tumor_probabilities) {
        std::cout << "\n=== ROC CURVE ANALYSIS ===" << std::endl;
        
        std::vector<ROCPoint> roc_points;
        
        // Test different thresholds
        for (double threshold = 0.1; threshold <= 0.9; threshold += 0.1) {
            ROCPoint point;
            point.threshold = threshold;
            
            int tp = 0, tn = 0, fp = 0, fn = 0;
            
            for (size_t i = 0; i < true_labels.size(); ++i) {
                int predicted = (tumor_probabilities[i] > threshold) ? 1 : 0;
                int actual = true_labels[i];
                
                if (actual == 1 && predicted == 1) tp++;
                else if (actual == 0 && predicted == 0) tn++;
                else if (actual == 0 && predicted == 1) fp++;
                else if (actual == 1 && predicted == 0) fn++;
            }
            
            point.sensitivity = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
            point.specificity = (tn + fp > 0) ? (double)tn / (tn + fp) : 0.0;
            point.false_positive_rate = 1.0 - point.specificity;
            
            roc_points.push_back(point);
        }
        
        // Print ROC table
        std::cout << "Threshold  Sensitivity  Specificity  FPR" << std::endl;
        std::cout << std::string(40, '-') << std::endl;
        
        for (const auto& point : roc_points) {
            std::cout << std::fixed << std::setprecision(1) << point.threshold 
                      << "        " << std::setprecision(3) << point.sensitivity
                      << "        " << std::setprecision(3) << point.specificity
                      << "        " << std::setprecision(3) << point.false_positive_rate
                      << std::endl;
        }
        
        // Calculate AUC (approximate)
        double auc = calculate_auc(roc_points);
        std::cout << "\nArea Under Curve (AUC): " << std::fixed << std::setprecision(3) << auc << std::endl;
        
        if (auc >= 0.9) {
            std::cout << "✓ EXCELLENT discriminative ability" << std::endl;
        } else if (auc >= 0.8) {
            std::cout << "✓ GOOD discriminative ability" << std::endl;
        } else if (auc >= 0.7) {
            std::cout << "○ FAIR discriminative ability" << std::endl;
        } else {
            std::cout << "⚠ POOR discriminative ability - needs improvement" << std::endl;
        }
    }
    
    // Demonstrate medical interpretation workflow
    static void demonstrate_medical_workflow(MedicalCNN& cnn, const ClinicalDataset& dataset) {
        std::cout << "\n=== MEDICAL INTERPRETATION WORKFLOW ===" << std::endl;
        
        // Select a few representative cases
        std::vector<int> demo_indices = {0, 1, static_cast<int>(dataset.mri_images.size()/2), static_cast<int>(dataset.mri_images.size()-1)};
        
        for (int idx : demo_indices) {
            if (idx >= dataset.mri_images.size()) continue;
            
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "CASE: " << dataset.patient_ids[idx] << std::endl;
            std::cout << "GROUND TRUTH: " << dataset.diagnoses[idx] << std::endl;
            std::cout << std::string(60, '-') << std::endl;
            
            // Step 1: Show original MRI
            MedicalImageProcessor processor(64, true);
            MedicalImageProcessor::MRIImage demo_mri(64, 64, dataset.labels[idx] == 1);
            demo_mri = processor.create_brain_mri(64, 64, 
                                                dataset.labels[idx] == 1 ? "tumor" : "normal");
            
            std::cout << "\nStep 1: MRI Slice Visualization" << std::endl;
            processor.visualize_mri_ascii(demo_mri, true);
            
            // Step 2: CNN Analysis
            std::cout << "\nStep 2: CNN Analysis" << std::endl;
            cnn.set_verbose(true);
            MedicalCNN::MedicalPrediction prediction = cnn.predict(dataset.mri_images[idx]);
            
            // Step 3: Activation Heatmap
            std::cout << "\nStep 3: Attention Heatmap (Areas of Interest)" << std::endl;
            Tensor heatmap = cnn.get_activation_heatmap();
            processor.create_activation_heatmap(demo_mri, heatmap, "CNN Attention Map");
            
            // Step 4: Clinical Assessment
            std::cout << "\nStep 4: Clinical Assessment" << std::endl;
            std::cout << "CNN Diagnosis: " << prediction.diagnosis << std::endl;
            std::cout << "Confidence: " << std::fixed << std::setprecision(1) 
                      << (prediction.confidence * 100) << "%" << std::endl;
            
            bool correct = (prediction.predicted_class == dataset.labels[idx]);
            std::cout << "Diagnostic Accuracy: " << (correct ? "✓ CORRECT" : "✗ INCORRECT") << std::endl;
            
            if (!correct) {
                std::cout << "⚠ CLINICAL NOTE: This case requires human expert review" << std::endl;
            }
        }
    }

private:
    static void print_case_by_case_analysis(const ClinicalDataset& dataset, 
                                           const std::vector<MedicalCNN::MedicalPrediction>& predictions) {
        std::cout << "\n=== CASE-BY-CASE ANALYSIS ===" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        std::cout << std::setw(12) << "Patient ID" 
                  << std::setw(15) << "Actual" 
                  << std::setw(15) << "Predicted" 
                  << std::setw(12) << "Confidence" 
                  << std::setw(10) << "Result" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (size_t i = 0; i < dataset.patient_ids.size(); ++i) {
            std::string actual = (dataset.labels[i] == 1) ? "TUMOR" : "NORMAL";
            std::string predicted = (predictions[i].predicted_class == 1) ? "TUMOR" : "NORMAL";
            bool correct = (predictions[i].predicted_class == dataset.labels[i]);
            std::string result = correct ? "✓" : "✗";
            
            std::cout << std::setw(12) << dataset.patient_ids[i]
                      << std::setw(15) << actual
                      << std::setw(15) << predicted
                      << std::setw(10) << std::fixed << std::setprecision(1) 
                      << (predictions[i].confidence * 100) << "%"
                      << std::setw(10) << result << std::endl;
        }
        std::cout << std::string(80, '-') << std::endl;
    }
    
    static double calculate_auc(const std::vector<ROCPoint>& roc_points) {
        if (roc_points.size() < 2) return 0.0;
        
        double auc = 0.0;
        for (size_t i = 1; i < roc_points.size(); ++i) {
            double width = roc_points[i].false_positive_rate - roc_points[i-1].false_positive_rate;
            double height = (roc_points[i].sensitivity + roc_points[i-1].sensitivity) / 2.0;
            auc += width * height;
        }
        
        return auc;
    }
    
    static void provide_clinical_recommendations(const MedicalCNN::TrainingMetrics& metrics) {
        std::cout << "\n=== CLINICAL RECOMMENDATIONS ===" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        std::cout << "DEPLOYMENT READINESS ASSESSMENT:" << std::endl;
        
        // Overall readiness
        if (metrics.accuracy >= 0.9 && metrics.sensitivity >= 0.9 && metrics.specificity >= 0.85) {
            std::cout << "✓ READY for clinical pilot testing" << std::endl;
            std::cout << "  - High accuracy and sensitivity achieved" << std::endl;
            std::cout << "  - Acceptable false positive rate" << std::endl;
        } else if (metrics.accuracy >= 0.8 && metrics.sensitivity >= 0.8) {
            std::cout << "○ PROMISING but needs refinement" << std::endl;
            std::cout << "  - Consider additional training data" << std::endl;
            std::cout << "  - Tune hyperparameters for better performance" << std::endl;
        } else {
            std::cout << "⚠ NOT READY for clinical use" << std::endl;
            std::cout << "  - Significant improvements needed" << std::endl;
            std::cout << "  - More training data and model optimization required" << std::endl;
        }
        
        std::cout << "\nCLINICAL INTEGRATION GUIDELINES:" << std::endl;
        std::cout << "1. Always use as a SECOND OPINION tool" << std::endl;
        std::cout << "2. Radiologist review required for all cases" << std::endl;
        std::cout << "3. High confidence predictions can expedite workflow" << std::endl;
        std::cout << "4. Low confidence cases need immediate expert review" << std::endl;
        
        if (metrics.sensitivity < 0.9) {
            std::cout << "\n⚠ CRITICAL: Sensitivity below 90%" << std::endl;
            std::cout << "   Risk of missing tumors - implement safeguards" << std::endl;
        }
        
        if (metrics.specificity < 0.8) {
            std::cout << "\n⚠ WARNING: High false positive rate" << std::endl;
            std::cout << "   May cause unnecessary anxiety and procedures" << std::endl;
        }
        
        std::cout << std::string(50, '=') << std::endl;
    }
};