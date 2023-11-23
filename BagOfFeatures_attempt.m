%% Εκκαθάριση
clear('all');clc;close('all');

% Σε περίπτωση που χρειαστεί να χρησιμοποιήσουμε την βιβλιοθήκη της vlfeat
% τρέχουμε την παρακάτω εντολή: 
% run('E:\Nik_Vas\vlfeat-0.9.21\toolbox\vl_setup.m')

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
load("C:\Users\vasil\OneDrive\Υπολογιστής\bag_words_demo\images\faces2\" + ...
    "ground_truth_locations.mat");

%% Preprocessing

% Διαχωρισμός δεδομένων σε εκπαίδευσης και αξιολόγησης

splitDatastore = splitEachLabel(datastore,1/5);
[Trainds,Testds] = splitEachLabel(datastore,0.7,"randomized");
trainlabelcount=countEachLabel(Trainds);

% Αλλαγή κλίμακας στα δεδομένα
target_x_scaling = 200;
for i = 1:length(Trainds.Files)

    img = read(Trainds);
    imshow(img)
    [imx,imy,imz] = size(img);
    scale_factor = target_x_scaling/imx;
    rescaled_train_images{i} = imresize(img,scale_factor,'bilinear');
    imshow(rescaled_train_images{i})
    rescaled_ground_truth_locations = gt_bounding_boxes{i}*scale_factor;
end

for i = 1:length(Testds.Files)

    img = read(Testds);
    imshow(img)
    [imx,imy,imz] = size(img);
    scale_factor = target_x_scaling/imx;
    rescaled_test_images{i} = imresize(img,scale_factor,'bilinear');
    imshow(rescaled_test_images{i})
    rescaled_ground_truth_locations = gt_bounding_boxes{i}*scale_factor;
end

    
grayTrainds = transform(Trainds,@im2gray);
grayTestds = transform(Testds,@im2gray);


%% Εξαγωγή χαρακτηριστικών χρησιμοποιώντας τον SIFT


for j=1:length(grayTrainds.UnderlyingDatastores{:}.Files)
    
    img=read(grayTrainds);
    features=extractSIFTfeatures(img); 
    train_image_features{j} = features;
    
end


for j=1:length(grayTestds.UnderlyingDatastores{:}.Files)
    
    img=read(grayTestds);
    features = extractSIFTfeatures(img);
    test_image_features{j} = features;
    
end

whos train_image_features test_image_features


%% Μαζεύουμε τα train δεδομένα από τα tall arrays
trainfeatureMatrix =[];
for j = 1:length(train_image_features)
    
    trainfeatureMatrix = [trainfeatureMatrix;
                          train_image_features{j}]; %#ok<*AGROW>
    
end

%% Αρχικοποίηση των παραμέτρων για τον k-means

TrainingLabels = grayTrainds.UnderlyingDatastores{:}.Labels;
TestingLabels = grayTestds.UnderlyingDatastores{:}.Labels;
classifier = cell(numel(unique(TestingLabels),1));
CVMdl = cell(numel(unique(TestingLabels),1));
genError = zeros(numel(unique(TestingLabels),1));

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

    numCentroids = 30;
    
    [~,C] = kmeans(gpuArray(trainfeatureMatrix),numCentroids, ...
        'MaxIter',10000,'Replicates',10);
    


    %% Αναπαράσταση του training και testing dataset (Coding)

    TrainingBoF = [];

    for j = 1:length(train_image_features)
        [~,train_gray_index] = pdist2(C,train_image_features{j}, ...
            'euclidean','Smallest',1);
        N = histcounts(train_gray_index, numCentroids);
        % Προσοχή! Για να προκύψουν τα τελικά ποσοστά για το BoF χρειάζεται να
        % διαιρούμε με το νούμερο των keypoints ανά εικόνα!

        train_vec_image = N./length(train_gray_index);
        TrainingBoF = [TrainingBoF;train_vec_image];

    end

    % Αναπαράσταση του testing dataset (Coding)

    TestingBoF = [];

    for j = 1:length(test_image_features)
        [~,test_gray_index] = pdist2(C,test_image_features{j}, ...
            'euclidean','Smallest',1);
        N = histcounts(test_gray_index, numCentroids);
        % Προσοχή! Για να προκύψουν τα τελικά ποσοστά για το BoF χρειάζεται να
        % διαιρούμε με το νούμερο των keypoints ανά εικόνα!

        test_vec_image = N./length(test_gray_index);
        TestingBoF = [TestingBoF;test_vec_image];

    end
    
    %% Εκπαιδεύοντας έναν ταξινομητή

    % Δημιουργούμε έναν error correcting output code ανάμεσα στο Bag of
    % Features των δεδομένων εκπαίδευσης και των labels που έχει

    classifier = fitcecoc(gather(TrainingBoF),TrainingLabels,'OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions',struct('MaxTime',1000));
    %%
    CVMdl{i} = crossval(classifier{i});
    genError(i) = kfoldLoss(CVMdl{i});
    %%
    % Δημιουργούμε προβλέψεις ανάμεσα στον classifier που δημιουργήσαμε
    % παραπάνω και τον table του test set
    [predictedLabels, scores]= predict(classifier,TestingBoF);

    % Αξιολόγηση

    confusionMatrix = confusionmat(TestingLabels,predictedLabels);

    % Υπολογισμός ακρίβειας
    accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));

    % Υπολογισμός Ακρίβειας, Recall και F1-score για την κάθε κλάση
    numClasses = size(confusionMatrix, 1);
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1Score = zeros(numClasses, 1);
    for j = 1:numClasses
        precision(j) = confusionMatrix(j,j) / sum(confusionMatrix(:,j));
        recall(j) = confusionMatrix(j,j) / sum(confusionMatrix(j,:));
        f1Score(j) = 2 * (precision(j) * recall(j)) / (precision(j) + ...
            recall(j));
    end

    fprintf('Current center >> %d\n',numCentroids);

% end

[row,~] = find(accuracy>=0.95 && accuracy<=0.99);
fprintf('Best accuracy found:%d',accuracy(row))
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
 bag = bagOfFeatures(Trainds,'GridStep',[24,24],'BlockWidth',[32 48 64 ...
     96 128]);
 % predFeatures = encode(bag,Trainds);
 % 
 % predFeaturesTest = encode(bag,Testds);

 
 categoryClassifier = trainImageCategoryClassifier(Trainds,bag);
 confMatrix = evaluate(categoryClassifier,Trainds);
 % classifier = fitcauto(predFeatures,Trainds.Labels,"Learners","all", ...
 %     "OptimizeHyperparameters","auto","HyperparameterOptimizationOptions", ...
 %     struct('UseParalle',true,'MaxTime',1000));

%% Συνάρτηση εξαγωγης χαρακτηριστικών με SIFT
function [features,featureMetrics,location]=extractSIFTfeatures(img)
    img = im2double(img);
    img = im2gray(img);
    points = detectSIFTFeatures(img,"NumLayersInOctave",5);
    [features,validPoints] = extractFeatures(img,points);
     featureMetrics=validPoints.Metric;location=validPoints.Location;
end
