function imo = cnn_imagenet_get_batch(images, varargin)
opts.imageSize = [227, 227] ;
opts.border = [29, 29] ;
opts.keepAspect = true ;
opts.numAugments = 1 ;
opts.transformation = 'none' ;
opts.averageImage = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 4 ;
opts.prefetch = true ;
opts = vl_argparse(opts, varargin);

% deal with border
if numel(opts.border)==1
    opts.border = [opts.border opts.border];
end
% average Image shape
if numel(opts.averageImage) == 3
    opts.averageImage = reshape(opts.averageImage, 1,1,3,1) ;
end
% the resize required
resize_vec = opts.imageSize(1:2)+opts.border;
% fetch if input a cell of image paths
fetch = numel(images) >= 1 && ischar(images{1}) ;
% prefetch if this is a prefetch call (and fetch)
prefetch = fetch & opts.prefetch ;

global prefetchOut hasPrefetched
if isempty(hasPrefetched) || ~existsOnGPU(prefetchOut)
    % initialize when program starts, or GPU is reset (indicates by
    % prefetchOut is unavailable)
    hasPrefetched = 0; 
end
if ~hasPrefetched
    prefetchOut = zeros(resize_vec(1), resize_vec(2), 3, ...
                        numel(images), 'single', 'gpuArray');
end

% 1 element for central crop, 2 elements for no crop and resize.
isCentralCrop = resize_vec(1); 

if prefetch
    hasPrefetched = 1;
    vl_imreadjpeg_gpu(images, prefetchOut, 'resize', isCentralCrop,...
        'numThreads', opts.numThreads, 'prefetch') ;
    imo = [] ;
    return ;
end
if fetch
    hasPrefetched = 0;
    im = vl_imreadjpeg_gpu(images, prefetchOut, 'resize', isCentralCrop,...
        'numThreads', opts.numThreads) ;
    
    st = floor(opts.border / 2)+1;
    ed = st + opts.imageSize(1:2) -1;
    prefetchOut = prefetchOut(st(1):ed(1), st(2):ed(2), :, :);
    
    imo = bsxfun(@minus, prefetchOut, gpuArray(opts.averageImage));    
else
    imo = images ;
end
