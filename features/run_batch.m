function[] = run_batch()

% ==================================================================
%
% The is the top-level file for computing features.  All the
% configuration information shuold go here.  Keep magic numbers out of
% all other files.
%
% ==================================================================

% add caffe matlab interface files to path (this is currently set to
% the installation directory on muir)
addpath('/usr2/caffe/matlab/caffe');

% points to image database base directory
source_dir = '/usr1/ksingh1/dataset/outside/input_visual_data/';

% points to feature output base directory
output_dir = '/usr1/ksingh1/dataset/outside/features_scene_L7/';

% set to 1 if you want to recompute features that already exist
params.force_recompute = 0;

% individual images are extracted from video at 10 fps, so a frame
% skip of 2 yields a computation of features at 5 fps
params.frame_skip = 2;

% what feature to compute
params.feature_type = 'L7';

% how to preprocess each image
params.image_prep_mode = 'center_crop';

% whether cafe should use the gpu
caffe_params.use_gpu = 0;

% ==================================================================


dir_list = {};
dir_count = 0;

listing = dir(source_dir);

for i=1:size(listing)
    name = listing(i).name;

    % Find all directories containing images
    % directories containing images will have an 'images' postfix.
    name_postfix = name(max([1, size(name)-5]): end);
    if ( (listing(i).isdir == 1) && (strcmp(name_postfix, 'images') == 1) )
        dir_count = dir_count+1;
        dir_list{dir_count} = name;
    end
end

fprintf('Found %d image directories\n', size(dir_list,2));

% compute deep features for contents of each folder
total_images_processed = 0;
for i=1:size(dir_list,2)
    input_dir = strcat(source_dir, '/', dir_list{i});
    output_file = strcat(output_dir, '/', dir_list{i});
    
    num_processed = compute_features(input_dir, output_file, params, caffe_params);
    total_images_processed = total_images_processed + num_processed;
end

fprintf('Done. Processed %d images.\n', total_images_processed);

