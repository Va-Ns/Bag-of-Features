function Indices = getindices(Trainds,Testds)

function Indices = getindices(Trainds,Testds)
%% GETINDICES Function Overview:
% This function extracts numerical indices from the filenames of images in
% training and testing datasets. It is designed to work with MATLAB's
% ImageDatastore objects. The function assumes that the filenames contain
% numerical identifiers following the pattern 'image_X', where X is the
% numerical ID. Additionally, it adjusts the indices based on the folder
% names, specifically adding 100 to the indices of images stored in a
% folder named 'faces2'.

%% Inputs:
%    Trainds: An ImageDatastore object containing the training dataset.
%    Testds: An ImageDatastore object containing the testing dataset.

%% Outputs:
%    Indices: A structure with two fields, Train_Indices and Test_Indices,
%    each containing arrays of numerical indices extracted and adjusted
%    from the filenames of the images in the respective datasets.

%% Processing Steps:
% 1. Iterate through each file in the training dataset:
%    a. Extract the file path and decompose it to get the filename and the
%       parent folder name.
%    b. Extract the numerical part of the filename, assumed to follow the
%       pattern 'image_X'.
%    c. If the parent folder name is 'faces2', add 100 to the numerical
%       identifier.
%    d. Store the adjusted numerical identifier in the Train_Indices array.
% 2. Repeat the process for each file in the testing dataset, storing the
%    results in the Test_Indices array.

%% Example Usage:
% Assuming Trainds and Testds are already defined ImageDatastore objects:
% Indices = getindices(Trainds, Testds);
% This will return a structure with adjusted numerical indices for both training and testing images.
    
    for i = 1:length(Trainds.Files)
        
        filePath = Trainds.Files{i};
        [filepath,fileName,~] = fileparts(filePath);
        imageNumberStr = extractAfter(fileName, 'image_');
        folderName = extractAfter(filepath, 'images\');
        imageNumber = str2double(imageNumberStr);
        if strcmp(folderName, 'faces2')
            imageNumber = imageNumber + 100;
        end
        Indices.Train_Indices(i) = imageNumber;
    end

    for i = 1:length(Testds.Files)
        filePath = Testds.Files{i};
        [filepath,fileName,~] = fileparts(filePath);
        imageNumberStr = extractAfter(fileName, 'image_');
        folderName = extractAfter(filepath, 'images\');
        imageNumber = str2double(imageNumberStr);
        if strcmp(folderName, 'faces2')
            imageNumber = imageNumber + 100;
        end
        Indices.Test_Indices(i) = imageNumber;
    end
    
end
