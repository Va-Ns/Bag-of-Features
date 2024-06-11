%% Clean up

clear;clc;close('all');


% Set a seed for reproducibility of the experiment
s = rng(1);

mkdir RUN_DIR\Codebook
mkdir RUN_DIR\SIFT_features_of_interest_points
mkdir RUN_DIR\Quantized_vector_descriptors

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

Gray_resized_datastore = Edge_Sampling_Vasilakis(Datastore,XScale);

total_time = toc; 
fprintf('\nFinished running interest point operator\n');

fprintf('Total number of images: %d, mean time per image: %f secs\n', numel(Datastore.Files), ...
                                                                total_time/numel(Datastore.Files));

%% Feature extraction using SIFT

load RUN_DIR\interest_points\interest_points.mat
reset(Gray_resized_datastore)
tic;
features = cell(1,length(Datastore.Files));

for i = 1:length(Datastore.Files)

    im = read(Gray_resized_datastore);
    [features{i},validPoints{i}] = extractFeatures(im,interest_points{i},"Method","SIFT");

end

total_time=toc;


save('RUN_DIR\SIFT_features_of_interest_points\SIFT_features.mat','features');
save('RUN_DIR\SIFT_features_of_interest_points\validPoints.mat','validPoints');

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

[~,Codebook,sse] = kmeans(gpuArray(double(descriptors)),300,"MaxIter",10, "Replicates",10);

save('RUN_DIR\Codebook\Codebook.mat', 'Codebook');
save('RUN_DIR\Codebook\SSE.mat', 'sse');

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
save("RUN_DIR\Quantized_vector_descriptors\training_descriptors_vq.mat",'training_descriptors_vq');
save("RUN_DIR\Quantized_vector_descriptors\testing_descriptors_vq.mat",'testing_descriptors_vq');
    
%% Training a Classifier

classifier = fitcauto(training_descriptors_vq,Trainds.Labels, ...
    'OptimizeHyperparameters','all','HyperparameterOptimizationOptions', ...
    struct('MaxObjectiveEvaluations',100,'Kfold',10));
[predictedLabels, scores]= predict(classifier,testing_descriptors_vq);

%% Evaluating the Classifier

confusionMatrix = confusionmat(Testds.Labels,predictedLabels);
Accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:))

%% Απεικόνιση του Bag of Words 

bar(training_descriptors_vq(1,:),'r'); 
xticks(1:10:size(Codebook,1));
xticklabels; % Rotate x-axis labels by 90 degrees
ytickformat("percentage")
