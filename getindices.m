function Indices = getindices(Trainds,Testds)
    
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
