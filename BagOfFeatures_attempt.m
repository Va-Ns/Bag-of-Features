%% Εκκαθάριση

clear;clc;close('all');
mkdir RUN_DIR\ Codebook
mkdir RUN_DIR\ SIFT_features_of_interest_points
mkdir RUN_DIR\ Quantized_vector_descriptors

%% Αρχικοποίηση της parpool
% delete(gcp)
% maxWorkers = maxNumCompThreads;
% disp("Maximum number of workers: " + maxWorkers);
% pool=parpool(maxWorkers/2);

%% Φόρτωσε τα στοιχεία από το Directory και δημιούργησε το datastore

fileLocation = uigetdir();
datastore=imageDatastore(fileLocation,"IncludeSubfolders",true,"LabelSource", ...
    "foldernames");
initialLabels = countEachLabel(datastore);
%load("C:\Users\vasil\OneDrive\Υπολογιστής\bag_words_demo\images\faces2\ground_truth_locations.mat");
%datastore.Labels = renamecats(datastore.Labels,["Background" "Faces"]);

%% Preprocessing

% Θέτουμε ένα seed για καλή επαναληψιμότητα του πειράματος
s = rng(1);

% Διαχωρισμός των δεδομένων σε εκπαίδευσης και ελέγχου με ποσοστό 70-30
[Trainds,Testds] = splitEachLabel(datastore,0.7,'randomized');
trainlabelcount=countEachLabel(Trainds);
testlabelcount=countEachLabel(Testds);

% Αρχικοποίηση της τιμής κατά την οποία θα γίνει η κλιμάκωση και
% προσδιορισμός του άξονα στο οποίο θα εφαρμοστεί.
XScale = 200;

% Μετατροπή των datastores ώστε να περιλαμβάνουν grayscale εικόνες
grayTrainds = transform(Trainds,@im2gray); 
grayTestds = transform(Testds,@im2gray);

% Αλλαγή του μεγέθους των grayscale εικόνων, με κλιμάκωση του x άξονα στα
% 200 pixel και χρήση της imresize για εφαρμογή του bilinear interpolation
% με σκοπό αυτόματη προσαρμογή του y άξονα.
grayresizedTrainds = transform(grayTrainds,@(x) imresize(x,XScale/size(x,1),'bilinear'));
grayresizedTestds = transform(grayTestds,@(y) imresize(y,XScale/size(y,1),'bilinear'));

% Προκειμένου να μπορέσουμε να εκτελέσουμε τον αλγόριθμο εντοπισμού των σημείων ενδιαφέροντος, θα 
% χρειαστεί να αλλάξουμε directory ώστε να μπορέσει να εκτελεστεί ένα συγκεκριμένο executable. 
% Για ευκολία δίνετε η δυνατότητα στον χρήστη να επιλέξει χειροκίνητα το μέρος όπου έχει αποθηκεύσει
% την συνάρτηση Edge_Sampling.m

getcurrentDirectory = pwd; 
EdgeSamplingLocation = uigetdir();

if isequal(getcurrentDirectory,EdgeSamplingLocation)

else   
    
    cd(EdgeSamplingLocation);
    
end
tic;

gray_resized_datastore = Edge_Sampling_Vasilakis(datastore,XScale);

total_time=toc; fprintf('\nFinished running interest point operator\n');

fprintf('Total number of images: %d, mean time per image: %f secs\n', numel(datastore.Files), ...
                                                                total_time/numel(datastore.Files));



%% Εξαγωγή χαρακτηριστικών χρησιμοποιώντας τον SIFT


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


%% Σχηματισμός του Λεξικού (Codebook Formation)

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
    
    [~,index] = pdist2(Codebook,double(features{Indices.Train_Indices(i)}), ...
                                        'euclidean','Smallest',1);
    N = histcounts(index, size(Codebook,1));
    
    % Προσοχή! Για να προκύψουν τα τελικά ποσοστά για το BoF χρειάζεται να διαιρούμε με το νούμερο 
    % των keypoints ανά εικόνα!

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

%% k-means clustering

% Η μεταβλητή idx αναμένεται να επιστρέψει σε ποια κέντρα
% κατηγοριοποιήθηκε η κάθε παρατήρηση. Γι' αυτό κι όλας παρατηρούμε ότι οι
% διαστάσεις της idx είναι οι ίδιες με των δεδομένων εκπαίδευσης.

% Η μεταβλητή C επιστρέφει την τοποθεσία των κεντροειδών. Η τοποθεσία του
% κάθε κέντρου στην idx αντικατοπτρίζεται από την σειρά στην μεταβλητή C.
% Για παράδειγμα η τοποθεσία του πρώτου κέντρου, idx(1), στον πολυδιάστατο
% χώρο των χαρακτηριστικών, είναι πρακτικά η πρώτη σειρά της C, δηλαδή
% C(1,:).

% Η μεταβλητή sumd επιστρέφει τα within-cluster αθροίσματα των αποστάσεων
% των παρατηρήσεων από τα κέντρα τους στα οποία έχουν ανατεθεί. Ένα πολύ
% σημαντικό μέτρο, καθώς όσο μικρότερο είναι το άθροισμα τόσο καλύτερα
% έχουν ανατεθεί τα δεδομένα στα κέντρα τους.

% Μεγάλη προσοχή όμως! Η within-cluster μετρική επηρεάζεται άμεσα από το
% πλήθος των παρατηρήσεων, δηλαδή τον αριθμό των δεδομένων που έχουμε.
% Όσο το πλήθος αυξάνεται, τόσο μεγαλύτερα θα βγαίνουν τα αθροίσματα των
% των αποστάσεων. Αυτό σημαίνει λοιπόν ότι η WCSS (Winthin-Cluster Sum of
% Squares) δεν συχνά συγκρίσιμη μεταξύ των clusters με διαφορετικά νούμερα
% παρατηρήσεων. Για να συγκρίνουμε τις within-cluster διαφοροποιήσει διαφο
% ρετικών clusters χρειαζόμαστε την μέση απόσταση ανά κέντρο.

% Η μεταβλητή D επιστρέφει τις αποστάσεις από κάθε σημείο προς κάθε κέντρο.
% for i=1:numel(unique(grayTrainds.UnderlyingDatastores{:}.Labels))

    



    
%% Εκπαιδεύοντας έναν ταξινομητή

classifier = fitcauto(training_descriptors_vq,Trainds.Labels, ...
    'OptimizeHyperparameters','all','HyperparameterOptimizationOptions', ...
    struct('MaxObjectiveEvaluations',100,'Kfold',10));
[predictedLabels, scores]= predict(classifier,testing_descriptors_vq);

%% Αξιολόγηση Ταξινομητή

confusionMatrix = confusionmat(Testds.Labels,predictedLabels)
Accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:))

%% Απεικόνιση του Bag of Words για τη πρώτη εικόνα 

% bar(TrainingBoF(1,:),'EdgeColor','red'); xticks(1:numCentroids);
% xticklabels(Labels); ytickformat("percentage")

%% Απεικόνιση του χώρου των χαρακτηριστικών

% Assume X_train is your training data and X_test is your testing data
% Assume idx_train is the cluster assignments for the training data
% Assume C is the centroids of the clusters
% Assume idx_test is the cluster assignments for the testing data

% % Use t-SNE to reduce dimensions
% X_train_tsne = tsne(trainfeatureMatrix);
% X_test_tsne = tsne(testfeatureMatrix);
% C_tsne = tsne(assignments);
%
% % Create a scatter plot for the training data
% gscatter(X_train_tsne(:,1), X_train_tsne(:,2), Labels, numCentroids);
% hold on;
%
% % Create a scatter plot for the centroids
% scatter(C_tsne(:,1), C_tsne(:,2), 100, 'k', 'x');
% hold on;
%
% % Create a scatter plot for the testing data
% gscatter(X_test_tsne(:,1), X_test_tsne(:,2),Labels, numCentroids);
% hold off;
%
% % Add a colorbar and title for your plot
% colorbar;
% title('t-SNE visualization of training data, centroids, and testing data');
%
% bag = bagOfFeatures(Trainds,'GridStep',[24,24],'BlockWidth',[32 48 64 ...
%     96 128]);
% predFeatures = encode(bag,Trainds);
% 
% predFeaturesTest = encode(bag,Testds);
%
% 
% categoryClassifier = trainImageCategoryClassifier(Trainds,bag);
% confMatrix = evaluate(categoryClassifier,Trainds);
% classifier = fitcauto(predFeatures,Trainds.Labels,"Learners","all", ...
%     "OptimizeHyperparameters","auto","HyperparameterOptimizationOptions", ...
%     struct('UseParalle',true,'MaxTime',1000));