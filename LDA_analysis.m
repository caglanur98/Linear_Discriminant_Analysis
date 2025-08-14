%% LINEAR DISCRIMINANT ANALYSIS 

% Created by      : Cagla Nur Demirkan
% Course          : Probabilistic and Statistical Modelling II 
% Lecturer        : Prof. Dr. Ryszard Auksztulewicz
% Semester        : Summer Semester 2023
%==========================================================================
% Description: This script performs Linear Discriminant Analysis (LDA) on 
% fMRI data using the Statistical Parametric Mapping (SPM) toolbox. 
% The script is designed to loop through a list of subjects. For each 
% subject, it applies PCA-based dimensionality reduction to data with many 
% features from one ROI, followed by tests for assumptions of multivariate 
% normality and equal covariance matrices. Then it performs LDA with cross-
% validation. The significance of the classification is evaluated using a 
% permutation test. Then, the decision boundaries between the 'stimulation'
% and 'imagery' conditions are visualized for each subject. The script 
% concludes by performing a one-sample t-test to determine if the group-level 
% classification accuracy is significantly different from chance level.
%==========================================================================


%% Initilization

src_path = '/Users/caglademirkan/Documents/MATLAB_Stat/Decoding_project/data_decoding'; % main directory where all subject data is stored
roi_path = fullfile(src_path, 'rois'); % roi directory that involves 10 roi files.

subjects = {'sub-001','sub-002','sub-003','sub-004','sub-005','sub-006','sub-007','sub-008','sub-009','sub-010'}; 

spm_path = '/Users/caglademirkan/Documents/MATLAB_NMDA/spm12/';

addpath(spm_path, src_path);
spm('Defaults','fmri');


%% Loop through each subject

alpha = 0.05; % actual significance level
n_subjects = length(subjects); % number of subjects (tests)
adjusted_alpha = alpha / n_subjects; %% adjusted p-value using Bonferroni correction

for subj_idx = 1:length(subjects)
    subject_id = subjects{subj_idx};
    disp(['Processing data for subject: ', subject_id]);

    % Step1: Defining paths for the current subject
    subj_path = fullfile(src_path, subject_id);
    glm_path = fullfile(subj_path, '1st_level_good_bad_Imag');
 
    beta_files = dir(fullfile(glm_path, 'beta_*.nii'));
    betas = arrayfun(@(x) fullfile(glm_path, x.name), beta_files, 'UniformOutput', false);

    % Step2: Data Preparation for the current subject
    % initialize empty arrays to store the indices for both conditions
    
    stim_indices = []; 
    imag_indices = [];


    for run = 1:6 % loop through each run
        start_idx = (run - 1) * 11; % 11 betas per run
        stim_indices = [stim_indices, start_idx + [1, 2, 3]];
        imag_indices = [imag_indices, start_idx + [4, 5, 6]];
        
    end

        stim_betas = betas(stim_indices); % extract the beta files for Stim condition
        imag_betas = betas(imag_indices); % extract the beta files for Imag condition

        roi_file = fullfile(roi_path, '1CONJ_right_BA2_uncorr.nii');

        Vroi = spm_vol(roi_file); % load the ROI mask
        roi_mask = spm_read_vols(Vroi);

        num_voxels_in_roi = sum(roi_mask(:) > 0); % determine the number of voxels in the ROI

        % Initialize arrays to store the extracted data
        stim_data_for_roi = zeros(length(stim_betas), num_voxels_in_roi);
        imag_data_for_roi = zeros(length(imag_betas), num_voxels_in_roi);

        % Loop through the beta images for the stim condition
        for i = 1:length(stim_betas)
            Vbeta = spm_vol(stim_betas{i});
            beta_data = spm_read_vols(Vbeta);
            masked_data = beta_data(roi_mask > 0); % mask the beta image with the ROI
            stim_data_for_roi(i, :) = masked_data; % store the voxel values directly
        end

       % Loop through the beta images for the imag condition
        for i = 1:length(imag_betas)
            Vbeta = spm_vol(imag_betas{i});
            beta_data = spm_read_vols(Vbeta);
            
            masked_data = beta_data(roi_mask > 0); % mask the beta image with the ROI
            imag_data_for_roi(i, :) = masked_data; % store the voxel values directly
  
        end

             combined_data = [stim_data_for_roi; imag_data_for_roi];
             labels = [ones(size(stim_data_for_roi, 1), 1); 2*ones(size(imag_data_for_roi, 1), 1)];

        
             % PCA for Dimensionality Reduction
             [coeff, score, ~] = pca(combined_data);
             reduced_data = score(:, 1:2); % use the first two principal components

             % Henze-Zirkler's Multivariate Normality Test (corrected)
             HZmvntest(reduced_data, adjusted_alpha);
             
             % Box s M test (corrected) for future LDA analysis
         
             grouped_data = [ones(18,1); 2*ones(18,1)];
             grouped_data = [grouped_data, reduced_data];
             MBoxtest(grouped_data, adjusted_alpha);
   
             % Step3: LDA with cross validation
             X = reduced_data;
             Y = labels;

             discr = fitcdiscr(X,Y,'KFold',5); 
             true = Y;
             predicted = kfoldPredict(discr);
             actual_accuracy = mean(true==predicted);
             disp(['LDA Classification Accuracy with CV: ', num2str(actual_accuracy*100), '%']);
    
             accuracies(subj_idx) = actual_accuracy;

             % Permutation Test for the current subject
             num_permutations = 1000;
             permuted_accuracies = zeros(1, num_permutations);

             
             for p = 1:num_permutations
                  shuffled_labels = Y(randperm(length(Y)));
                  discr_perm = fitcdiscr(X,shuffled_labels,'KFold',5);
                  predicted_perm = kfoldPredict(discr_perm);
                  permuted_accuracies(p) = mean(shuffled_labels==predicted_perm);
               
              end

                  p_values_perm(subj_idx) = mean(permuted_accuracies >= actual_accuracy);
                  disp(['Permutation test p-value for ', subject_id, ': ', num2str(p_values_perm(subj_idx))]);

                  % Step4: Visualization decision boundry for each subject

                    X= reduced_data;
                    Y= labels;

                    discr = fitcdiscr(X,Y);
                    true = Y;
                    predicted = predict(discr,X);
                    actual_accuracy2 = mean(true==predicted);

                    figure;
                    scatter(X(Y==1, 1), X(Y==1, 2), 'r', 'filled'); % stimulation condition
                    hold on;
                    scatter(X(Y==2, 1), X(Y==2, 2), 'b', 'filled'); % imagery condition
                    
                    K = discr.Coeffs(1,2).Const; % constant term of the decision boundary
                    L = discr.Coeffs(1,2).Linear; % linear coefficients of the decision boundary
                    f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
                    h2 = fimplicit(f, [min(X(:,1)) max(X(:,1)) min(X(:,2)) max(X(:,2))]);
                    h2.Color = 'k';
                    h2.LineWidth = 2;
                    
                    xlabel('Principal Component 1');
                    ylabel('Principal Component 2');
                    title(['LDA Decision Boundary for Subject: ', subject_id]);
                    legend('Stimulation', 'Imagery', 'Decision Boundary');
                    hold off;

end


%% One-sample t-test for group

mean_accuracy = mean(accuracies);
chance_level = 0.5;

[h, p, ci, stats] = ttest(accuracies, chance_level);

% Results
if h == 1
    disp(['The mean classification accuracy is significantly different from chance level. p-value = ', num2str(p)]);
else
    disp(['The mean classification accuracy is not significantly different from chance level. p-value = ', num2str(p)]);
end

%% Save relevant results

save('/Users/caglademirkan/Documents/MATLAB_Stat/Decoding_project/results/LDA_results.mat', 'adjusted_alpha', 'accuracies', 'p_values_perm','mean_accuracy', 'h','p', 'ci', 'stats');



