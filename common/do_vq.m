function do_vq(config_file)

%% Function takes the regions in interest_points directory, along with
%% their descirptors and vector-quantizes them using the codebook
%% specified in the VQ structure.
  
%% The VQ label is then stored in the interest_point/interest_xxxx.mat
%% file in the descriptor_vq variable. A histogram over codebook
%% entries is also computed and stored.

%% Before running this, you must have run:
%%    do_random_indices - to generate random_indices.mat file
%%    do_preprocessing - to get the images that the operator will run on  
%%    do_interest_op  - to get extract interest points (x,y,scale) from each image
%%    do_representation - to get appearance descriptors of the regions 
  
%% You must also have either: (a) run do_form_codebook to generate a codebook file
%%                        or: (b) already have a valid codebook file in
%% CODEBOOK_DIR, matching the VQ.Codebook_Type tag and of size VQ.Codebook_Size.
  
%% R.Fergus (fergus@csail.mit.edu) 03/10/05.  
   
  
        
%% Evaluate global configuration file
eval('config_file_1');

%% If no VQ structure specifying codebook
%% give some defaults

if ~exist('VQ')
  %% use default codebook family
  VQ.Codebook_Type = 'generic';
  %% 1000 words is standard setting
  VQ.Codebook_Size = 1000;
end

 
%% Get list of interest point file names
% ip_file_names =  genFileNames({Global.Interest_Dir_Name}, ...
%     1:Categories.Total_Frames,RUN_DIR,Global.Interest_File_Name, ...
%     '.mat',Global.Num_Zeros);
 
%% How many images are we processing?
nImages = Categories.Total_Frames;

%% Now load up codebook
%codebook_name = [CODEBOOK_DIR , '/', VQ.Codebook_Type ,'_', num2str(VQ.Codebook_Size) , '.mat'];
load('RUN_DIR\Codebook\Codebook.mat');
mkdir RUN_DIR\ Quantized_vector_descriptors

%%% Load up interest point file
load("RUN_DIR\SIFT_features_of_interest_points\SIFT_features.mat")

tic;
  
  %%% Loop over all images....
  for i=1:nImages
    
    if (mod(i,10)==0)
      fprintf('.%d',i);
    end
    

    %% Find number of points per image
    %nPoints = length(features{i});
    
    %% Set distance matrix to all be large values
    %distance = Inf * ones(nPoints,size(centers,1)); %VQ.Codebook_Size
    
    %% Loop over all centers and all points and get L2 norm btw. the two.
    % for p = 1:nPoints
    %   for c = 1:size(centers,1)
    %     distance(p,c) = norm(centers(:,c) - double(interest_points{:,p}));
    %   end
    % end
    

    
    %% Now find the closest center for each point
    %[tmp,descriptor_vq] = min(distance,[],2);

    %% Now compute histogram over codebook entries for image
    %histogram = zeros(1,VQ.Codebook_Size);
    
    % for p = 1:nPoints
    %   histogram(descriptor_vq(p)) = histogram(descriptor_vq(p)) + 1;
    % end
    % 
    % %%% transpose to match other variables
    % descriptor_vq = descriptor_vq';


    [~,index] = pdist2(centers,double(features{i}), ...
        'euclidean','Smallest',1);
    N = histcounts(index, size(centers,1));
    descriptors_vq{i}= N./length(index);
    

    
    %% append descriptor_vq variable to file....
    save("RUN_DIR\Quantized_vector_descriptors\descriptors_vq.mat", ...
        'descriptors_vq');
    
  end

total_time=toc;

A = cellfun(@sum,descriptors_vq);

if abs(A-1) <1e-10

    fprintf('\nThe histogram procedure was completed correctly\n')

else

    fprintf('Oopsie')

end

train_vq_descriptors= [];

for i = 1:length(Categories.All_Train_Frames)
    X1 = descriptors_vq{Categories.All_Train_Frames(i)};
    train_vq_descriptors = [train_vq_descriptors;X1];
end

test_vq_descriptors= [];

for i = 1:length(Categories.All_Test_Frames)
    X1 = descriptors_vq{Categories.All_Test_Frames(i)};
    test_vq_descriptors = [test_vq_descriptors;X1];
end
fprintf('\nFinished running VQ process\n');
fprintf('Total number of images: %d, mean time per image: %f secs\n', ...
    nImages,total_time/nImages);

save("RUN_DIR\Quantized_vector_descriptors\train_vq_descriptors.mat", ...
    "train_vq_descriptors");

save("RUN_DIR\Quantized_vector_descriptors\test_vq_descriptors.mat", ...
    "test_vq_descriptors");

end