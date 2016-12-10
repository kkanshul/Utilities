function [scores] = deep_features(im, mode)
% scores = matcaffe_demo(im, use_gpu)
%
% Demo of the matlab wrapper using the ILSVRC network.
%
% input
%   im       color image as uint8 HxWx3
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000-dimensional ILSVRC score vector
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-5.5/lib64
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  im = imread('../../examples/cat.jpg');
%  scores = matcaffe_demo(im, 1);
%  [score, class] = max(scores);


% uncomment for classification score (eightth layer)
% model_def_file = '../../models/bvlc_reference_caffenet/deploy.prototxt';

input_data = {prepare_image(im,mode)};

% do forward pass to get scores
scores = caffe('forward', input_data);

% ------------------------------------------------------------------------
function images = prepare_image(im, mode)
% ------------------------------------------------------------------------
%d = load('ilsvrc_2012_mean');
d=load('/usr1/ksingh1/models/hybridCNN/hybridCNN_mean.binaryproto');
IMAGE_MEAN = d.image_mean;
CROPPED_DIM = 227;

% resize to fixed input size
im = single(im);

% permute from RGB to BGR (IMAGE_MEAN is already BGR)
im = im(:,:,[3 2 1]) - IMAGE_MEAN;

if (mode == 'center_crop')
    images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 1, 'single');
    
    % take the center crop
    images(:,:,:,1) = permute(im(15:15+226, 15:15+226, :), [2 1 3]);
else
    fprintf('WARNING: UNSUPPORTED MODE to prepare_image! Returning blank image!!!\n');
    images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 1, 'single');
end
