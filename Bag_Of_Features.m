%% Clean up

clear;clc;close('all');

% Set a seed for reproducibility of the experiment
s = rng(1);

mkdir Workspace\Codebook
mkdir Workspace\SIFT_features_of_interest_points
mkdir Workspace\Quantized_vector_descriptors
mkdir Workspace\SIFT_features_of_interest_points

%% Initiate the parpool in case you need man power
% delete(gcp)
% maxWorkers = maxNumCompThreads;
% disp("Maximum number of workers: " + maxWorkers);
% pool = parpool(maxWorkers/2);

%% Load the data from the Directory and create the Datastore
fileLocation = uigetdir();
Datastore = imageDatastore(fileLocation,"IncludeSubfolders",true,"LabelSource","foldernames");
initialLabels = countEachLabel(Datastore);

%% Preprocessing

% Initialize the value by which the scaling will be performed and specify the axis to which it will 
% be applied.

XScale = 200;

% In order to be able to run the interest point detection algorithm, we will need to change the 
% directory so that a specific executable can be executed. For convenience, the user is given the 
% option to manually select the place where the Edge_Sampling.m function is stored.

getcurrentDirectory = pwd; 
EdgeSamplingLocation = uigetdir();

if isequal(getcurrentDirectory,EdgeSamplingLocation)

else   
    
    cd(EdgeSamplingLocation);
    
end
tic;

[Gray_resized_datastore,Variables] = Edge_Sampling_VN(Datastore,XScale,"WorkspaceDir", ...
                                                                [getcurrentDirectory,'\Workspace'], ...
                                                                "Show",true);

total_time = toc; 
fprintf('\nFinished running interest point operator\n');

fprintf('Total number of images: %d, mean time per image: %f secs\n', numel(Datastore.Files), ...
                                                                total_time/numel(Datastore.Files));

%% Feature extraction using SIFT

reset(Gray_resized_datastore)
tic;
features = cell(1,length(Datastore.Files));

for i = 1:length(Datastore.Files)

    im = read(Gray_resized_datastore);
    [features{i},validPoints{i}] = extractFeatures(im,Variables.interest_points{i},"Method","SIFT");

end

total_time=toc;

cd(getcurrentDirectory)
save('Workspace\SIFT_features_of_interest_points\SIFT_features.mat','features');
save('Workspace\SIFT_features_of_interest_points\validPoints.mat','validPoints');

fprintf('\nFinished running descriptor operator\n');

fprintf('Total number of images: %d, mean time per image: %f secs\n', length(Datastore.Files), ...
    total_time/length(Datastore.Files));


%% Codebook Formation

[Trainds,Testds] = splitEachLabel(Gray_resized_datastore.UnderlyingDatastores{:},0.75,'randomized');


Indices = getindices(Trainds,Testds);
descriptors = [];
for i = 1:length(Indices.Train_Indices)

    descriptors =[descriptors; features{Indices.Train_Indices(i)}];

end

[~,Codebook,~] = kmeans(gpuArray(double(descriptors)),300,"MaxIter",10,"Replicates",10);

Codebookfilepath = fullfile([getcurrentDirectory,'\Workspace\Codebook'],'Codebook.mat'); 
save(Codebookfilepath, 'Codebook');

training_descriptors_vq = zeros(length(Indices.Train_Indices),size(Codebook,1));
testing_descriptors_vq = zeros(length(Indices.Test_Indices),size(Codebook,1));

for i=1:length(Indices.Train_Indices)
        
    fprintf('Currently at training image:%d\n',i);
    
    [~,index] = pdist2(Codebook,double(features{Indices.Train_Indices(i)}),'euclidean','Smallest',1);
    N = histcounts(index, size(Codebook,1));
    
    % Beware! To obtain the final percentages for BoF we need to divide by the number of keypoints 
    % per image!

    training_descriptors_vq(i,:)= N./length(index);
   
end

for i=1:length(Indices.Test_Indices)
        
    fprintf('Currently at testing image:%d\n',i);
    
    [~,index] = pdist2(Codebook,double(features{Indices.Test_Indices(i)}),'euclidean','Smallest',1);
    N = histcounts(index, size(Codebook,1));
    testing_descriptors_vq(i,:)= N./length(index);
   
end

TrainingVQDfilepath = fullfile([getcurrentDirectory,'\Workspace\Quantized_vector_descriptors'], ...
                                                                     'training_descriptors_vq.mat');
TestingVQDfilepath = fullfile([getcurrentDirectory,'\Workspace\Quantized_vector_descriptors'], ...
                                                                      'testing_descriptors_vq.mat');

save(TrainingVQDfilepath,'training_descriptors_vq');
save(TestingVQDfilepath,'testing_descriptors_vq');
    
%% Training a Classifier

classifier = fitcsvm(training_descriptors_vq,Trainds.Labels, 'OptimizeHyperparameters', ...
    'all','HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',100,'Kfold',10,...
    'Optimizer','gridsearch','NumGridDivisions',20));
[predictedLabels, scores]= predict(classifier,testing_descriptors_vq);

%% Evaluating the Classifier

confusionMatrix = confusionmat(Testds.Labels,predictedLabels);
Accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:))
%% Visualize the Bag of Visual Words for an image 

reset(Datastore)
% Define the index of the image to visualize
imageIndex = 1;

% Get the features of the first image
imageFeatures = double(features{Indices.Train_Indices(imageIndex)});

% Compute the nearest codebook centers for the features of the first image
[~, index] = pdist2(Codebook, imageFeatures, 'euclidean', 'Smallest', 1);

% Compute the histogram of the assignments
N = histcounts(index, size(Codebook, 1));

% Normalize the histogram to form the final vector representation
vectorRepresentation = N / length(index);

% Reduce the dimensionality of the codebook using PCA
[coeff, score, ~] = pca(Codebook);
reducedCodebook = score(:, 1:2);

% Create Voronoi cells for the reduced codebook
figure;
voronoi(reducedCodebook(:, 1), reducedCodebook(:, 2));
hold on;

% Highlight the centers that the image was assigned to
highlightedCenters = unique(index);
scatter(reducedCodebook(highlightedCenters, 1), reducedCodebook(highlightedCenters, 2), 100, 'r', 'filled');

% Set axis limits and labels
axis equal;
xlabel('PCA Component 1');
ylabel('PCA Component 2');
title('Voronoi Cells of Codebook with Highlighted Centers');
hold off;

% Load and display the initial image
figure;
subplot(1, 2, 1);
imageFile = Datastore.Files{Indices.Train_Indices(imageIndex)};
image = imread(imageFile);
imshow(image);
title('Initial Image');

% Plot the histogram representation and highlight the centers
subplot(1, 2, 2);
bar(vectorRepresentation, 'r');
hold on;
highlightedValues = vectorRepresentation(highlightedCenters);
bar(highlightedCenters, highlightedValues, 'b');
xticks(1:10:size(Codebook, 1));
xlabel("Codebook components");
ylabel("Percentage of participation for every center");
title("Histogram Representation with Highlighted Centers");
legend('All Centers', 'Assigned Centers');
hold off;

%% Depict the vector of 5 images using t-SNE
% Define the number of images to visualize
numImages = 5;
% Initialize a matrix to store the vector representations
vectorRepresentations = zeros(numImages, size(Codebook, 1));
% Loop through the first numImages images
for imageIndex = 1:numImages
    % Get the features of the image
    imageFeatures = double(features{Indices.Train_Indices(imageIndex)});
    % Compute the nearest codebook centers for the features of the image
    [~, index] = pdist2(Codebook, imageFeatures, 'euclidean', 'Smallest', 1);
    % Compute the histogram of the assignments
    N = histcounts(index, size(Codebook, 1));
    % Normalize the histogram to form the final vector representation
    vectorRepresentations(imageIndex, :) = N / length(index);
end
% Use t-SNE on the vectorRepresentations to reduce them to 3D
reducedVectorRepresentations = tsne(vectorRepresentations, 'NumDimensions', 3);
% Plot the vector representations in a 3D space
figure('Color', 'w'); % Set background color to white
set(gcf, 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]); % Resize the figure
hold on;
colormap(parula(numImages)); % Use a colorful colormap
scatter3(reducedVectorRepresentations(:, 1), reducedVectorRepresentations(:, 2), reducedVectorRepresentations(:, 3), 100, 1:numImages, 'filled');
textColor = 'k'; % Set the text color for all vectors
for i = 1:numImages
    % Plot the vector representation
    quiver3(0, 0, 0, reducedVectorRepresentations(i, 1), reducedVectorRepresentations(i, 2), reducedVectorRepresentations(i, 3), 'Color', colors(i, :), 'LineWidth', 2);
    % Place the text label in a distinct location
    textX = reducedVectorRepresentations(i, 1) * 1.1;
    textY = reducedVectorRepresentations(i, 2) * 1.1;
    textZ = reducedVectorRepresentations(i, 3) * 1.1;
    text(textX, textY, textZ, sprintf('Image %d', i), 'FontSize', 12, 'Color', textColor, 'HorizontalAlignment', 'center');
end
xlabel('t-SNE Dimension 1', 'FontSize', 14);
ylabel('t-SNE Dimension 2', 'FontSize', 14);
zlabel('t-SNE Dimension 3', 'FontSize', 14);
title('3D Vector Representations of the First 5 Images using t-SNE', 'FontSize', 16);
grid on;
view(3); % Set the default 3D view
box on;
% Add a light source and adjust the lighting
camlight('headlight');
lighting('gouraud');
hold off;