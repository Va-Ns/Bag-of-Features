function [gray_resized_datastore,Variables] = Edge_Sampling_VN(imageDatastore,XScale,Options,Plot)

%% Edge Sampling Operator:
% This function implements an edge sampling operator that:
%    1. Converts images to grayscale and resizes them based on a specified scale.
%    2. Detects edges in the images using the Canny edge detector.
%    3. Samples a specified maximum number of points (edgels) from the detected edges, optionally weighted by edge strength.
%    4. Assigns a scale to each sampled point by drawing from a uniform distribution over a specified range.
%    5. Optionally plots the results for visualization.

%% Inputs:
%    imageDatastore: An ImageDatastore object containing the images to be processed.
%    XScale: The target scale to resize the images to. The resizing is proportional to maintain aspect ratio.
%    Options: A structure containing the following fields:
%       - Max_Points: The maximum number of points to sample from the detected edges.
%       - Scale: A range [min, max] from which to draw the scale for each sampled point.
%       - Weighted_Sampling: A flag indicating whether to weight the sampling by edge strength.
%       - WorkspaceDir: The directory where output variables will be saved.
%    Plot: A structure containing the following field:
%       - Show: A flag indicating whether to plot the results.

%% Outputs:
%    gray_resized_datastore: An ImageDatastore object containing the resized grayscale images.
%    Variables: A structure containing the following fields:
%       - x: The x coordinates of the sampled points.
%       - y: The y coordinates of the sampled points.
%       - scale: The scales assigned to the sampled points.
%       - score: The edge strengths of the sampled points.
%       - interest_points: A matrix containing the x and y coordinates of the sampled points for each image.

%% Disclaimer!

% A big part of this code snippet is formed based on the Edge_Sampling.m file created by R.Fergus 
% (fergus@csail.mit.edu) at the ICCV of 2005 (03/10/05)



arguments
    imageDatastore            {mustBeUnderlyingType(imageDatastore, ...
                                                              "matlab.io.datastore.ImageDatastore")}
    XScale                    {mustBePositive,mustBeInteger,mustBeNonmissing} = 200
    Options.Max_Points        {mustBePositive,mustBeInteger,mustBeNonmissing} = 200
    Options.Scale             (1,:)                                           = 10:30;
    Options.Weighted_Sampling {mustBePositive,mustBeInteger,mustBeNonmissing} = 1
    Options.WorkspaceDir           {mustBeText} 
    Plot.Show                 {mustBeNumericOrLogical}                        = false
end
       
   

%%% Get total number of images
nImages = length(imageDatastore.Files);

%%% Loop over all images

x = []; xx = [];
y = []; yy = [];
strength = [];
scale = []; score = [];

gray_datastore = transform(imageDatastore,@im2gray); 
gray_resized_datastore = transform(gray_datastore, @(z) imresize(z, XScale/size(z,1),'bilinear'));

for i = 1:nImages

    % Reset variables
    % x = []; xx = [];
    % y = []; yy = [];
    % strength = [];
    % scale = []; score = [];

    % read in image

    % The function requires the quality and size transformed images. So to avoid having to join the 
    % individual datastores together, we transform the original datastore by following the same 
    % procedures as the main file.


    im = read(gray_resized_datastore);

    % Find canny edges using Oxford VGG code
    curves = vgg_xcv_segment(uint8(im),'canny_edges');


    % Concatenate all edgel segments together into one big array
    for b = 1 : length(curves)
        
        % x location    
        xx = [ xx , curves{b}(1,:)];  %#ok<AGROW>
        
        % y location
        yy = [ yy , curves{b}(2,:)];  %#ok<AGROW>

        % edge strength
        strength = [ strength , curves{b}(3,:)];  %#ok<AGROW>

    end

    % Total number of edge pixels exracted from image
    nEdgels = length(strength);


    if nEdgels > 0 %% check that some edgels were found in the image

        % Obtain sampling density
        % choose btw. uniform and weighted towards those edgels with a
        % stronger edge strength
        if Options.Weighted_Sampling

            sample_density = strength / sum(strength);

        else

            sample_density = ones(1,nPoints)/nPoints;

        end

        % Choose how many points to sample
        nPoints_to_Sample = Options.Max_Points;

        % draw samples from density
        samples = discrete_sampler(sample_density,nPoints_to_Sample,1);

        % Lookup points corresponding to samples
        x{i} = xx(samples); %#ok<AGROW>
        y{i} = yy(samples); %#ok<AGROW>
        interest_points{i} =[x{i}',y{i}']; %#ok<AGROW>

        % now draw scales from uniform
        scale{i} = rand(1,nPoints_to_Sample)*(max(Options.Scale)- ...
                                               min(Options.Scale))+min(Options.Scale); %#ok<AGROW>

        % get scores for each points (its edge strength)
        score{i} = strength(samples); %#ok<AGROW>

    else % No edgels found in image at allInterest_Point.Weighted_Sampling    = 1;


        % Set all output variables for the frame to be empty
        x = [];
        y = [];
        scale = [];
        score = [];

    end


    if Plot.Show

    % Plotting of the results so the user can check the operation of the function 

        % Show image with edgels marked
        figure; %#ok<UNRCH>
        clf;

        imagesc(im);
        colormap(gray);
        hold on;
        plot(xx,yy,'m.','MarkerSize',8)
        title('Raw edgels');

        % Show image with region marked
        figure;
        clf;

        imagesc(im);
        colormap(gray);
        hold on;

        % for b=1:length(scale)
        plot(x{i},y{i},'b.','MarkerSize',10);

        %drawcircle(y{b},x{b},scale{b}*2+1,'g',1);
        viscircles(interest_points{i}, scale{i}/2,'Color','g');
        title(['Interest regions on image: ',num2str(i)]);

    end

end


fprintf('Image: %d from: %d\n',i,nImages);

% output_file_names{i} = strjoin(output_file_names{i},'');
% save('RUN_DIR\interest_points\output_file_names.mat','output_file_names');
Variables.x = x;
Variables.y = y;
Variables.scale = scale;
Variables.score = score;
Variables.interest_points = interest_points;

xfilepath = fullfile(Options.WorkspaceDir,'x.mat');
yfilepath = fullfile(Options.WorkspaceDir,'y.mat');
scalefilepath = fullfile(Options.WorkspaceDir,'scale.mat');
scorefilepath = fullfile(Options.WorkspaceDir,'score.mat');
interest_pointsfilepath = fullfile(Options.WorkspaceDir,'interest_points.mat');

save(xfilepath,'x');
save(yfilepath,'y');
save(scalefilepath,'scale');
save(scorefilepath,'score');
save(interest_pointsfilepath,'interest_points');


end
