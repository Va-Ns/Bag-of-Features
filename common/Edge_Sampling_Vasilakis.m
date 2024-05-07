function gray_resized_datastore = Edge_Sampling_Vasilakis(image_file_names,XScale,Interest_Point)

  %% Simple interest operator that:
  %%    1. Runs Canny edge detector on image
  %%    2. Sample Interest_Point.Max_Points points from set of edgels, weighted according to their intensity
  %%    3. For each sample, set scale by drawing from uniform distribution ...
  %%        over Interest_Point.Scale
    
  %% Inputs: 
  %%      1. image_file_names - cell array of filenames of all images to be processed
  %%      2. output_file_names - cell array of output filenames
  %%      3. Interest_Point - structure holding all settings of the interest operator
  
  %% Outputs:
  %%      None - it saves the results for each image to the files
  %%      specified in output_file_names.
  %%      Each file holds 4 variables:
  %%          x - x coordinates of points (1 x Interest_Point.Max_Points)
  %%          y - y coordinates of points (1 x Interest_Point.Max_Points)
  %%          scale - characteristic scale of points (radius, in pixels)  (1 x Interest_Point.Max_Points)
  %%          score - importance measure of each point, determined by edge strength of pixels (1 x Interest_Point.Max_Points).

    
  %%% R.Fergus (fergus@csail.mit.edu)  03/10/05.  
     
    
%%% DEBUG switch. Normally set to 0 but if set to >=1 will start plotting
%%% out results so user can check operation of the fucntion    
eval('config_file_1')
DEBUG       = 0

%%% Default parameters section
if (nargin<3)
   fprintf('Edge_Sampling: No settings specified, so using defaults...\n');
   Interest_Points.Max_Points = 200;
   Interest_Points.Weighted_Sampling = 1;
   Interest_Points.Weighted_Scale = 1;
end

%%% Get total number of images
nImages = length(image_file_names.Files);

%%% Loop over all images

x = []; xx = [];
y = []; yy = [];
strength = [];
scale = []; score = [];

gray_datastore = transform(image_file_names,@im2gray); 
gray_resized_datastore = transform(gray_datastore, @(z) imresize(z, ...
      XScale/size(z,1),'bilinear'));

for i = 1:nImages

    % Reset variables
    % x = []; xx = [];
    % y = []; yy = [];
    % strength = [];
    % scale = []; score = [];

    % read in image

    % Η συνάρτηση απαιτεί τις μετασχηματισμένες κατά ποιότητα και μέγεθος
    % εικόνες. Για να μην χρειαστεί λοιπόν να ενώσουμε τα επιμέρους
    % datastores μεταξύ τους, μετασχηματίζουμε το αρχικό datastore
    % ακολουθώντας τις διαδικασίες που γίνονται και στο κεντρικό αρχείο.


    im=read(gray_resized_datastore);
    %imshow(im)
    % Get size
    [imx,imy,imz]=size(im);

    % Convert to grayscale if not already so...
    if (imz>1)
        im=rgb2gray(im);
    end

    % Find canny edges using Oxford VGG code
    curves=vgg_xcv_segment(uint8(im),'canny_edges');


    % Concatenate all edgel segments together into one big array
    for b=1:length(curves)
        xx = [ xx , curves{b}(1,:)]; %% x location
        yy = [ yy , curves{b}(2,:)]; %% y location
        strength = [ strength , curves{b}(3,:)]; %% edge strength
    end

    % Total number of edge pixels exracted from image
    nEdgels = length(strength);


    if (nEdgels>0) %% check that some edgels were found in the image

        % Obtain sampling density
        % choose btw. uniform and weighted towards those edgels with a
        % stronger edge strength
        if Interest_Point.Weighted_Sampling
            sample_density = strength / sum(strength);
        else
            sample_density = ones(1,nPoints)/nPoints;
        end

        % Choose how many points to sample
        nPoints_to_Sample = Interest_Point.Max_Points;

        % draw samples from density
        samples = discrete_sampler(sample_density,nPoints_to_Sample,1);

        % Lookup points corresponding to samples
        x{i} = xx(samples);
        y{i} = yy(samples);
        interest_points{i} =[x{i}',y{i}'];
        % now draw scales from uniform
        scale{i} = rand(1,nPoints_to_Sample)*(max(Interest_Point.Scale)-min(Interest_Point.Scale))+min(Interest_Point.Scale);

        % get scores for each points (its edge strength)
        score{i} = strength(samples);

    else % No edgels found in image at allInterest_Point.Weighted_Sampling    = 1;


        % Set all output variables for the frame to be empty
        x = [];
        y = [];
        scale = [];
        score = [];

    end


    if DEBUG


        % Show image with edgels marked
        figure(1); clf;
        imagesc(im);
        colormap(gray);
        hold on;
        plot(xx,yy,'m.','MarkerSize',8)
        title('Raw edgels');

        % Show image with region marked
        figure(2); clf;
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
save('RUN_DIR\interest_points\x.mat','x');
save('RUN_DIR\interest_points\y.mat','y');
save('RUN_DIR\interest_points\scale.mat','scale');
save('RUN_DIR\interest_points\score.mat','score');
save('RUN_DIR\interest_points\interest_points.mat','interest_points');


end
