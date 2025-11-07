#pragma once
#include "../neural_network/utils/Tensor.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>

class MedicalImageProcessor {
public:
    struct MRIImage {
        std::vector<std::vector<double>> data;
        int width, height;
        bool has_tumor;
        std::string patient_id;
        
        MRIImage(int w, int h, bool tumor = false) 
            : width(w), height(h), has_tumor(tumor) {
            data.resize(height, std::vector<double>(width, 0.0));
        }
    };
    
    struct MedicalStats {
        double mean;
        double std_dev;
        double min_val;
        double max_val;
    };

private:
    bool verbose;
    int target_size;
    double normalization_mean;
    double normalization_std;
    
public:
    MedicalImageProcessor(int size = 64, bool verb = false) 
        : target_size(size), verbose(verb), normalization_mean(0.5), normalization_std(0.5) {}
    
    // Convert grayscale MRI image to normalized tensor
    Tensor mri_to_tensor(const MRIImage& mri, bool normalize = true) {
        if (verbose) {
            std::cout << "Converting MRI to tensor (size: " << mri.width << "x" << mri.height << ")" << std::endl;
        }
        
        // Resize if needed
        MRIImage resized_mri = resize_mri(mri, target_size, target_size);
        
        // Create tensor [1, H, W] for grayscale
        Tensor tensor({1, target_size, target_size}, false);
        
        // Copy and normalize pixel values
        for (int i = 0; i < target_size; ++i) {
            for (int j = 0; j < target_size; ++j) {
                double pixel_val = resized_mri.data[i][j];
                
                if (normalize) {
                    // X[i,j] = I[i,j] / 255.0
                    pixel_val = pixel_val / 255.0;
                    
                    // Normalize: X' = (X - μ) / σ
                    pixel_val = (pixel_val - normalization_mean) / normalization_std;
                }
                
                tensor(0, i, j) = pixel_val;
            }
        }
        
        return tensor;
    }
    
    // Create synthetic brain MRI slices for demonstration
    MRIImage create_brain_mri(int width, int height, const std::string& type) {
        MRIImage mri(width, height);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> brain_tissue(120.0, 20.0);  // Gray matter intensity
        std::normal_distribution<> csf(40.0, 10.0);            // Cerebrospinal fluid
        std::normal_distribution<> white_matter(160.0, 15.0);   // White matter
        std::normal_distribution<> tumor_tissue(200.0, 25.0);   // Tumor (hyperintense)
        
        // Create brain anatomy
        int center_x = width / 2;
        int center_y = height / 2;
        double brain_radius = std::min(width, height) * 0.4;
        
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                double dist_center = std::sqrt((i - center_y) * (i - center_y) + 
                                             (j - center_x) * (j - center_x));
                
                if (dist_center > brain_radius) {
                    // Background (skull, air)
                    mri.data[i][j] = std::max(0.0, std::min(255.0, brain_tissue(gen) * 0.3));
                } else if (dist_center > brain_radius * 0.8) {
                    // Gray matter (cortex)
                    mri.data[i][j] = std::max(0.0, std::min(255.0, brain_tissue(gen)));
                } else if (dist_center < brain_radius * 0.3) {
                    // CSF (ventricles)
                    mri.data[i][j] = std::max(0.0, std::min(255.0, csf(gen)));
                } else {
                    // White matter
                    mri.data[i][j] = std::max(0.0, std::min(255.0, white_matter(gen)));
                }
            }
        }
        
        // Add specific pathology based on type
        if (type == "tumor") {
            add_tumor(mri, tumor_tissue, gen);
            mri.has_tumor = true;
        } else if (type == "edema") {
            add_edema(mri, brain_tissue, gen);
            mri.has_tumor = true;  // Edema often indicates tumor
        } else if (type == "normal") {
            mri.has_tumor = false;
        } else if (type == "metastasis") {
            add_multiple_tumors(mri, tumor_tissue, gen);
            mri.has_tumor = true;
        }
        
        return mri;
    }
    
    // Compute medical image statistics
    MedicalStats compute_stats(const MRIImage& mri) {
        MedicalStats stats;
        std::vector<double> all_pixels;
        
        for (const auto& row : mri.data) {
            for (double pixel : row) {
                all_pixels.push_back(pixel);
            }
        }
        
        stats.min_val = *std::min_element(all_pixels.begin(), all_pixels.end());
        stats.max_val = *std::max_element(all_pixels.begin(), all_pixels.end());
        
        // Compute mean
        double sum = std::accumulate(all_pixels.begin(), all_pixels.end(), 0.0);
        stats.mean = sum / all_pixels.size();
        
        // Compute standard deviation
        double variance = 0.0;
        for (double pixel : all_pixels) {
            variance += (pixel - stats.mean) * (pixel - stats.mean);
        }
        stats.std_dev = std::sqrt(variance / all_pixels.size());
        
        return stats;
    }
    
    // Visualize MRI slice in ASCII
    void visualize_mri_ascii(const MRIImage& mri, bool show_tumor_info = true) {
        std::cout << "\n=== Brain MRI Slice Visualization ===" << std::endl;
        if (show_tumor_info) {
            std::cout << "Tumor Status: " << (mri.has_tumor ? "POSITIVE" : "NEGATIVE") << std::endl;
        }
        
        // Scale down for ASCII display
        int display_height = 20;
        int display_width = 40;
        
        for (int i = 0; i < display_height; ++i) {
            for (int j = 0; j < display_width; ++j) {
                // Map to original image coordinates
                int orig_i = (i * mri.height) / display_height;
                int orig_j = (j * mri.width) / display_width;
                
                double intensity = mri.data[orig_i][orig_j];
                char symbol;
                
                if (intensity < 50) symbol = ' ';       // Dark (CSF, background)
                else if (intensity < 100) symbol = '.'; // Low intensity
                else if (intensity < 150) symbol = 'o'; // Medium (gray matter)
                else if (intensity < 200) symbol = 'O'; // High (white matter)
                else symbol = '#';                       // Very high (tumor)
                
                std::cout << symbol;
            }
            std::cout << std::endl;
        }
        
        std::cout << "Legend: ' '=CSF/Background, '.'=Low, 'o'=Gray Matter, 'O'=White Matter, '#'=Tumor" << std::endl;
    }
    
    // Create activation heatmap overlay
    void create_activation_heatmap(const MRIImage& original, const Tensor& activation_map, 
                                 const std::string& title = "CNN Activation Heatmap") {
        std::cout << "\n=== " << title << " ===" << std::endl;
        std::cout << "Areas where CNN detected potential abnormalities:" << std::endl;
        
        int display_height = 15;
        int display_width = 30;
        
        // Find activation map range for normalization
        double min_activation = 1e6, max_activation = -1e6;
        for (int i = 0; i < activation_map.get_shape()[1]; ++i) {
            for (int j = 0; j < activation_map.get_shape()[2]; ++j) {
                double val = activation_map(0, i, j);
                min_activation = std::min(min_activation, val);
                max_activation = std::max(max_activation, val);
            }
        }
        
        for (int i = 0; i < display_height; ++i) {
            for (int j = 0; j < display_width; ++j) {
                // Map to activation coordinates
                int act_i = (i * activation_map.get_shape()[1]) / display_height;
                int act_j = (j * activation_map.get_shape()[2]) / display_width;
                
                double activation = activation_map(0, act_i, act_j);
                double normalized = (activation - min_activation) / (max_activation - min_activation);
                
                char symbol;
                if (normalized < 0.2) symbol = ' ';
                else if (normalized < 0.4) symbol = '.';
                else if (normalized < 0.6) symbol = '+';
                else if (normalized < 0.8) symbol = '*';
                else symbol = '#';
                
                std::cout << symbol;
            }
            std::cout << std::endl;
        }
        
        std::cout << "Heatmap: ' '=Low, '.'=Mild, '+'=Moderate, '*'=High, '#'=Very High Activation" << std::endl;
    }

private:
    MRIImage resize_mri(const MRIImage& original, int new_width, int new_height) {
        MRIImage resized(new_width, new_height, original.has_tumor);
        resized.patient_id = original.patient_id;
        
        double x_ratio = (double)original.width / new_width;
        double y_ratio = (double)original.height / new_height;
        
        for (int i = 0; i < new_height; ++i) {
            for (int j = 0; j < new_width; ++j) {
                // Bilinear interpolation
                int x = (int)(j * x_ratio);
                int y = (int)(i * y_ratio);
                
                x = std::max(0, std::min(x, original.width - 1));
                y = std::max(0, std::min(y, original.height - 1));
                
                resized.data[i][j] = original.data[y][x];
            }
        }
        
        return resized;
    }
    
    void add_tumor(MRIImage& mri, std::normal_distribution<>& tumor_dist, std::mt19937& gen) {
        // Add a single tumor (glioblastoma-like)
        int tumor_x = mri.width * 0.6;  // Typical location
        int tumor_y = mri.height * 0.4;
        int tumor_radius = std::min(mri.width, mri.height) * 0.15;
        
        for (int i = std::max(0, tumor_y - tumor_radius); 
             i < std::min(mri.height, tumor_y + tumor_radius); ++i) {
            for (int j = std::max(0, tumor_x - tumor_radius); 
                 j < std::min(mri.width, tumor_x + tumor_radius); ++j) {
                
                double dist = std::sqrt((i - tumor_y) * (i - tumor_y) + 
                                      (j - tumor_x) * (j - tumor_x));
                
                if (dist < tumor_radius) {
                    // Create irregular tumor boundary
                    double boundary_factor = 1.0 - (dist / tumor_radius);
                    boundary_factor *= (0.8 + 0.4 * ((double)rand() / RAND_MAX));
                    
                    if (boundary_factor > 0.3) {
                        mri.data[i][j] = std::max(0.0, std::min(255.0, tumor_dist(gen)));
                    }
                }
            }
        }
    }
    
    void add_edema(MRIImage& mri, std::normal_distribution<>& tissue_dist, std::mt19937& gen) {
        // Add perilesional edema (darker regions around lesion)
        int edema_x = mri.width * 0.3;
        int edema_y = mri.height * 0.6;
        int edema_radius = std::min(mri.width, mri.height) * 0.2;
        
        for (int i = std::max(0, edema_y - edema_radius); 
             i < std::min(mri.height, edema_y + edema_radius); ++i) {
            for (int j = std::max(0, edema_x - edema_radius); 
                 j < std::min(mri.width, edema_x + edema_radius); ++j) {
                
                double dist = std::sqrt((i - edema_y) * (i - edema_y) + 
                                      (j - edema_x) * (j - edema_x));
                
                if (dist < edema_radius) {
                    // Edema appears as hypointense (darker) regions
                    mri.data[i][j] = std::max(0.0, std::min(255.0, tissue_dist(gen) * 0.6));
                }
            }
        }
    }
    
    void add_multiple_tumors(MRIImage& mri, std::normal_distribution<>& tumor_dist, std::mt19937& gen) {
        // Add multiple small metastatic lesions
        std::vector<std::pair<int, int>> locations = {
            {mri.width * 0.2, mri.height * 0.3},
            {mri.width * 0.7, mri.height * 0.2},
            {mri.width * 0.8, mri.height * 0.7}
        };
        
        for (auto& loc : locations) {
            int tumor_radius = std::min(mri.width, mri.height) * 0.08;  // Smaller lesions
            
            for (int i = std::max(0, loc.second - tumor_radius); 
                 i < std::min(mri.height, loc.second + tumor_radius); ++i) {
                for (int j = std::max(0, loc.first - tumor_radius); 
                     j < std::min(mri.width, loc.first + tumor_radius); ++j) {
                    
                    double dist = std::sqrt((i - loc.second) * (i - loc.second) + 
                                          (j - loc.first) * (j - loc.first));
                    
                    if (dist < tumor_radius) {
                        mri.data[i][j] = std::max(0.0, std::min(255.0, tumor_dist(gen)));
                    }
                }
            }
        }
    }
};