
function[total_images_processed]=compute_features(input_dir, out_file, params, caffe_params)
%Input: 
% input_dir: folder containing images
% out_file: folder where we will save our .mat file
% params.force_recompute: if 1, always overwrite
% params.frame_skip: process every 'frame_skip' frames in the directory
% params.feature_type: type of feature to compute: e.g, L7, L8, ...
% caffe_params: various params to pass along to caffe

% =============================================================

full_output_path = strcat(out_file, '_', params.feature_type, '.mat');

% if output .mat file exists, and force_recompute has not been
% specified, exit immediately, there's nothing to do
if ( (params.force_recompute==0) && (exist(full_output_path) ~= 0) )
    total_images_processed = 0;
    return;
end

%model_file = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
%model_file = '../models/bvlc_reference_caffenet/static_new_fine_iter_6000.caffemodel';
model_file = '/usr1/ksingh1/models/hybridCNN/hybridCNN_iter_700000.caffemodel';

if (params.feature_type == 'L7')
    % deploy file for deep feature (seventh layer)
    %model_def_file = '../models/bvlc_reference_caffenet/deploy_feature.prototxt';
    model_def_file = '/usr1/ksingh1/models/hybridCNN/hybridCNN_deploy_FC7.prototxt';
    FEATURE_SIZE = 4096;
elseif (params.feature_type == 'L5')
    % deploy file for deep feature (seventh layer)
    model_def_file = '../models/bvlc_reference_caffenet/deploy_fifth.prototxt';
    FEATURE_SIZE = 9216;
else
    % uncomment for classification score (eighth layer)
    model_def_file = '../models/bvlc_reference_caffenet/deploy_finetune.prototxt';
    FEATURE_SIZE = 2;
end
    


% =============================================================

% initialize caffe, if necessary
if caffe('is_initialized') == 0
    caffe('init', model_def_file, model_file);
    if (caffe_params.use_gpu == 1)
        caffe('set_mode_gpu');
    else
        caffe('set_mode_cpu');
    end
    caffe('set_phase_test');
end

% get all images in the input folder
listing = dir(strcat(input_dir, '/*.jpg'));

image_list = {};
image_count = 0;

% build a list of all images that must be processed (note this
% takes into account frame_skip)
for i=1:params.frame_skip:size(listing)
    image_count = image_count+1;
    image_list{image_count} = listing(i).name;
end

fprintf('===============================================================\n');
fprintf('Processing %d images in directory: %s\n', image_count, input_dir);
fprintf('===============================================================\n');

% prealocate a large matrix to hold the results of feature extraction
% for all frames
all_features = zeros(image_count, FEATURE_SIZE, 'single');

% compute_deep features for each frame
start_time = tic;
last_time = tic;
for i=1:image_count

    % occasionally pretty-print a status message
    STATUS_REPORT_CHUNK = 500;
    if mod(i,STATUS_REPORT_CHUNK) == 0
        elapsed_time_ms = 1000 * toc(last_time) / STATUS_REPORT_CHUNK;
        remaining = (image_count - i) * elapsed_time_ms / 1000 / 60;
        fprintf(['Computed %d of %d in dir (%.2f ms/image, %.2f min remaining for dir)\n'], i, image_count, elapsed_time_ms, remaining);
        last_time = tic;
    end
        
    full_image_path = strcat(input_dir, '/' , image_list{i});
    im = imread(full_image_path);
    im = imresize(im, [256 256]);
    result = deep_features(im, params.image_prep_mode);
    %FIXME(kayvonf): not handling other image_prep_modes right now
    all_features(i,:) = result{1}(:);
    
end

elapsed_time = toc(start_time);
time_per_image = 1000 * elapsed_time / image_count;

fprintf(['===============================================================\n']);
fprintf('Directory complete: %.2f sec, %.2f ms/image\n', elapsed_time, time_per_image);
fprintf(['===============================================================\n']);

% save the features
% TODO(kayvonf): we shouldn't wait for the whole directory to be done
% to do this save.  This risks losing a lot of work.
save(full_output_path, 'all_features', 'image_list');

total_images_processed = image_count;

end
