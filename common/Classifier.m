clear;clc;eval('config_file_1')
load("RUN_DIR\Quantized_vector_descriptors\train_vq_descriptors.mat")
load("RUN_DIR\Quantized_vector_descriptors\test_vq_descriptors.mat")
datastore = imageDatastore("RUN_DIR\resized_images");
for i = 1:100
    face_labels{i} ='Face';
    background{i} = 'Background';
end
face_labels = categorical(face_labels);
background = categorical(background);
labels = [face_labels';background'];
datastore.Labels = labels;
TrainingLabels = datastore.Labels(Categories.All_Train_Frames);
TestingLabels = datastore.Labels(Categories.All_Test_Frames);

classifier = fitcauto(train_vq_descriptors,TrainingLabels, ...
    'OptimizeHyperparameters','all','HyperparameterOptimizationOptions', ...
    struct('MaxTime',1e9,'UseParallel',true,'Kfold',10, ...
    'MaxObjectiveEvaluations',200));

[predictedLabels, scores]= predict(classifier,test_vq_descriptors);
confusionMatrix = confusionmat(TestingLabels,predictedLabels);
accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));
