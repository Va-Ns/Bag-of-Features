accuracy = zeros(1,10);

for p = 1:10

    % Εκκαθάριση
    %clear('all');clc;close('all');
    mkdir RUN_DIR\ Codebook
    mkdir RUN_DIR\ SIFT_features_of_interest_points
    mkdir RUN_DIR\ Quantized_vector_descriptors

    % Φόρτωσε τα στοιχεία από το Directory και δημιούργησε το datastore
    %fileLocation = uigetdir();
    fileLocation ='C:\Users\vasil\MATLAB Drive\Thesis\Matlab Projects\Bag of Features\images';
    datastore=imageDatastore(fileLocation,"IncludeSubfolders",true,"LabelSource", ...
        "foldernames");
    initialLabels = countEachLabel(datastore);
    %load("C:\Users\vasil\OneDrive\Υπολογιστής\bag_words_demo\images\faces2\ground_truth_locations.mat");
    %datastore.Labels = renamecats(datastore.Labels,["Background" "Faces"]);


    % Θέτουμε ένα seed για καλή επαναληψιμότητα του πειράματος
    %s = rng(1);

    % Διαχωρισμός των δεδομένων σε εκπαίδευσης και ελέγχου με ποσοστό 70-30
    [Trainds,Testds] = splitEachLabel(datastore,0.7,'randomized');
    trainlabelcount=countEachLabel(Trainds);
    testlabelcount=countEachLabel(Testds);


    %message = checkimagesizes(Trainds,Testds);fprintf(message)
    % Αρχικοποίηση της τιμής κατά την οποία θα γίνει η κλιμάκωση και
    % προσδιορισμός του άξονα στο οποίο θα εφαρμοστεί.
    XScale = 200;

    % Μετατροπή των datastores ώστε να περιλαμβάνουν grayscale εικόνες
    grayTrainds = transform(Trainds,@im2gray);
    grayTestds = transform(Testds,@im2gray);

    % Αλλαγή του μεγέθους των grayscale εικόνων, με κλιμάκωση του x άξονα στα
    % 200 pixel και χρήση της imresize για εφαρμογή του bilinear interpolation
    % με σκοπόπο αυτόματη προσαρμογή του y άξονα.
    grayresizedTrainds = transform(grayTrainds,@(x) imresize(x,XScale/size(x,1),'bilinear'));
    grayresizedTestds = transform(grayTestds,@(y) imresize(y,XScale/size(y,1),'bilinear'));



    % Προκειμένου να μπορέσουμε να εκτελέσουμε τον αλγόριθμο εντοπισμού των
    % σημείων ενδιαφέροντος, θα χρειαστεί να αλλάξουμε directory ώστε να
    % μπορέσει να εκτελεστεί ένα συγκεκριμένο executable. Για ευκολία δίνετε η
    % δυνατότητα στον χρήστη να επιλέξει χειροκίνητα το μέρος όπου έχει
    % αποθηκεύσει την συνάρτηση Edge_Sampling.m
    %getcurrentDirectory = pwd;
    EdgeSamplingLocation =... %uigetdir();
    'C:\Users\vasil\MATLAB Drive\Thesis\Matlab Projects\Bag of Features\common';
    if isequal(getcurrentDirectory,EdgeSamplingLocation)

    else
        cd(EdgeSamplingLocation);

    end
    tic;

    gray_resized_datastore = Edge_Sampling_Vasilakis(datastore,XScale);
    total_time=toc; fprintf('\nFinished running interest point operator\n');
    fprintf('Total number of images: %d, mean time per image: %f secs\n', ...
        numel(datastore.Files),total_time/numel(datastore.Files));


    load RUN_DIR\interest_points\interest_points.mat
    reset(gray_resized_datastore)
    tic;
    features =cell(1,length(datastore.Files));
    for i = 1:length(datastore.Files)
        im =read(gray_resized_datastore);
        [features{i},validPoints{i}] = extractFeatures(im,interest_points{i},"Method","SIFT");
    end
    total_time=toc; clear im


    save('RUN_DIR\SIFT_features_of_interest_points\SIFT_features.mat', ...
        'features');
    save('RUN_DIR\SIFT_features_of_interest_points\validPoints.mat', ...
        'validPoints');

    fprintf('\nFinished running descriptor operator\n');
    fprintf('Total number of images: %d, mean time per image: %f secs\n', ...
        length(datastore.Files),total_time/length(datastore.Files));


    Indices = getindices(Trainds,Testds);
    descriptors = [];
    for i = 1:length(Indices.Train_Indices)

        descriptors =[descriptors; features{Indices.Train_Indices(i)}];

    end



    [~,Codebook,sse] = kmeans(double(descriptors),300,"MaxIter",10, ...
        "Replicates",10,'Options',statset('UseParallel',true));

    save('RUN_DIR\Codebook\Codebook.mat', 'Codebook');
    save('RUN_DIR\Codebook\SSE.mat', 'sse');

    training_descriptors_vq = zeros(length(Indices.Train_Indices),size(Codebook,1));
    testing_descriptors_vq = zeros(length(Indices.Test_Indices),size(Codebook,1));

    for i=1:length(Indices.Train_Indices)

        fprintf('Currently at training image:%d\n',i);

        [~,index] = pdist2(Codebook,double(features{Indices.Train_Indices(i)}), ...
            'euclidean','Smallest',1);
        N = histcounts(index, size(Codebook,1));
        training_descriptors_vq(i,:)= N./length(index);

    end

    for i=1:length(Indices.Test_Indices)

        fprintf('Currently at testing image:%d\n',i);

        [~,index] = pdist2(Codebook,double(features{Indices.Test_Indices(i)}), ...
            'euclidean','Smallest',1);
        N = histcounts(index, size(Codebook,1));
        testing_descriptors_vq(i,:)= N./length(index);

    end
    save("RUN_DIR\Quantized_vector_descriptors\training_descriptors_vq.mat", ...
        'training_descriptors_vq');
    save("RUN_DIR\Quantized_vector_descriptors\testing_descriptors_vq.mat", ...
        'testing_descriptors_vq');


    classifier = fitcauto(training_descriptors_vq,Trainds.Labels, ...
        'OptimizeHyperparameters','all','HyperparameterOptimizationOptions', ...
        struct('MaxTime',1e9,'UseParallel',true,'Kfold',10));
    [predictedLabels, scores]= predict(classifier,testing_descriptors_vq);

    confusionMatrix = confusionmat(Testds.Labels,predictedLabels);
    Accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));
    accuracy(p) = Accuracy;

end

