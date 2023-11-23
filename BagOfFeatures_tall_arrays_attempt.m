%% Εκκαθάριση
clear('all');clc;close('all');
%% Αρχικοποίηση της parpool
delete(gcp)
maxWorkers = maxNumCompThreads;
disp("Maximum number of workers: " + maxWorkers);
pool=parpool(maxWorkers/2);

%% Φόρτωσε τα στοιχεία από το Directory και δημιούργησε το datastore
fileLocation = uigetdir();
datastore=imageDatastore(fileLocation,"IncludeSubfolders",true,"LabelSource", ...
    "foldernames");
initialLabels = countEachLabel(datastore);

%% Διαχωρισμός δεδομένων σε εκπαίδευσης και αξιολόγησης και μετατροπή

splitDatastore = splitEachLabel(datastore,1/5);
[Trainds,Testds] = splitEachLabel(splitDatastore,0.8,"randomized");
trainlabelcount=countEachLabel(Trainds);

if ~isequal(initialLabels(:,1),trainlabelcount(:,1))
    
    error(['Training datastore is missing labels from ' ...
        'the original datastore.']);
    
else
    
    msgbox(['Training datastore is not missing labels from the ' ...
        'original datastore.'])
    
    
end

grayTrainds = transform(Trainds,@im2gray);
grayTestds = transform(Testds,@im2gray);


%% Εξαγωγή χαρακτηριστικών χρησιμοποιώντας τον SIFT

tic
parfor i=1:length(grayTrainds.UnderlyingDatastores{:}.Files)
    
    img=read(grayTrainds);
    features=extractSIFTfeatures(img); %#ok<PFTUSW>
    train_image_features{i} = tall(features);
    
end
trainparfortime=toc;


tic
parfor i=1:length(grayTestds.UnderlyingDatastores{:}.Files)
    
    img=read(grayTestds);
    features = extractSIFTfeatures(img); %#ok<PFTUSW>
    test_image_features{i} = tall(features);
    
end
testparfortime=toc;

whos train_image_features test_image_features


%% Μαζεύουμε τα train δεδομένα από τα tall arrays
trainfeatureMatrix =[];
for i = 1:length(train_image_features)
    
    trainfeatureMatrix = [trainfeatureMatrix;
        gather(train_image_features{i})]; %#ok<*AGROW>
    
end
min_train = min(trainfeatureMatrix,[],'all');
max_train = max(trainfeatureMatrix,[],'all');

normalizedtrainfeatureMatrix = (trainfeatureMatrix - min_train)/( ...
    max_train-min_train);


%% Αρχικοποίηση των παραμέτρων για τον k-means

Labels = unique(grayTrainds.UnderlyingDatastores{:}.Labels);

numCentroids = 69;
% numCentroids = numel(unique(datastore.Labels)); 

% options = statset('TolFun',1e-4);

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

tic
[~,C] = kmeans(gpuArray(normalizedtrainfeatureMatrix), ...
    numCentroids,'MaxIter',10000,'Replicates',10);
kmeansforloop = toc;


%% Αναπαράσταση του training και testing dataset (Coding)

TrainingBoFTall = [];

for i = 1:length(train_image_features)
    
    % Κανονικοποίηση της i-οστής εικόνας
    normalized_train_image_features{i} = (train_image_features{i} - ...
        min_train) / (max_train - min_train);

    [~,train_gray_index] = pdist2(C,gather(normalized_train_image_features ...
        {i}), 'euclidean','Smallest', 1); 
    N = histcounts(train_gray_index, numCentroids);
    % Προσοχή! Για να προκύψουν τα τελικά ποσοστά για το BoF χρειάζεται να
    % διαιρούμε με το νούμερο των keypoints ανά εικόνα!

    train_vec_image = N./length(train_gray_index);
    TrainingBoFTall = [TrainingBoFTall;train_vec_image];

end

% Αναπαράσταση του testing dataset (Coding)

TestingBoFTall = [];

for i = 1:length(test_image_features)
    
    % Κανονικοποίηση της i-οστής εικόνας
    normalized_test_image_features{i} = (test_image_features{i} - ...
        min_train) / (max_train - min_train);

    [~,test_gray_index] = pdist2(C,gather(normalized_test_image_features{i}), ...
        'euclidean','Smallest', 1); 
    N = histcounts(test_gray_index, numCentroids);
    % Προσοχή! Για να προκύψουν τα τελικά ποσοστά για το BoF χρειάζεται να
    % διαιρούμε με το νούμερο των keypoints ανά εικόνα!

    test_vec_image = N./length(test_gray_index);
    TestingBoFTall = [TestingBoFTall;test_vec_image];

end

%% Προετοιμασία των δεδομένων με την μετατροπή τους σε table
TrainingLabels = grayTrainds.UnderlyingDatastores{:}.Labels;
TestingLabels = grayTestds.UnderlyingDatastores{:}.Labels;
% classifier = cell(numel(unique(TestingLabels),1));
% CVMdl = cell(numel(unique(TestingLabels),1));
% genError = zeros(numel(unique(TestingLabels),1));
%% Εκπαιδεύοντας έναν ταξινομητή

% Δημιουργούμε έναν error correcting output code ανάμεσα στο Bag of
% Features των δεδομένων εκπαίδευσης και των labels που έχει


classifier = fitcecoc(TrainingBoFTall, TrainingLabels, ...
    'OptimizeHyperparameters','all',...
'HyperparameterOptimizationOptions',struct('MaxTime',1000, ...
'UseParallel',true));

% Δημιουργούμε προβλέψεις ανάμεσα στον classifier που δημιουργήσαμε
% παραπάνω και τον table του test set
[predictedLabels, scores]= predict(classifier,sortedTestLabels);

% CVMdl = crossval(classifier);
% genError = kfoldLoss(CVMdl)

% Αξιολόγηση

confusionMatrix = confusionmat(TestingLabels,predictedLabels);

% Υπολογισμός ακρίβειας 
accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));

% Υπολογισμός Ακρίβειας, Recall και F1-score για την κάθε κλάση
numClasses = size(confusionMatrix, 1);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);
for i = 1:numClasses
    precision(i) = confusionMatrix(i,i) / sum(confusionMatrix(:,i));
    recall(i) = confusionMatrix(i,i) / sum(confusionMatrix(i,:));
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end
%% Απεικόνιση του Bag of Words για τη πρώτη εικόνα 
bar(TrainingBoF(1,:),'EdgeColor','red'); xticks(1:numCentroids);
xticklabels(Labels); ytickformat("percentage")

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



%% Συνάρτηση εξαγωγης χαρακτηριστικών με SIFT
function features=extractSIFTfeatures(img)
    img = im2double(img);
    points = detectSIFTFeatures(img,"NumLayersInOctave",5);
    [features,~] = extractFeatures(img,points);
    % featureMetrics=validPoints.Metric;location=validPoints.Location;
end
