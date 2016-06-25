function net=add_le_sparse_conv(net, dim, lrs, stride, pad, nnz, useGPU, groups)
    % generate the zero locations
    % fix seed in order to be predictable
    
    opts=struct('modelType', 'lenet',...
        'whitenData', false,...
        'contrastNormalization', false,...
        'train', struct('gpus',1), ...
        'imdbPath', fullfile(vl_rootnn, 'data','cifar','imdb.mat'));
    % some values about:
    % learning rate, weightDecay, batchSize, momentum is in cnn_cifar_init
    % the sparse parameters include: nnz
    %%
    opts.train.gpus=1;
    tag = 'sparse-nnz_3_32_32_64_64';
    
    imdb = load(opts.imdbPath);
    opts.expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
    
    %Load the last epoch's result
    Epoch = 80;
    modelPath = fullfile(opts.expDir, sprintf('net-epoch-%d.mat', Epoch));
    netstruct = load(modelPath,'net', 'stats') ;
    originalnet = vl_simplenn_tidy(netstruct.net) ;
    originalnet = vl_simplenn_move(originalnet, 'gpu') ;
    im = gpuArray(imdb.images.data(:,:,:,10001:10100));
    res = vl_simplenn(originalnet, im);
    
    
    %We can do some analysis on the outputs of relu units
    layer_idx = numel(net.layers) + 1;
    res_local = gather(res(layer_idx).x);
    
    pairwiseMetric = @covKernel;
    weightMargin = 0;
    kernelGraph = pairwiseMetric(res_local);
    [V,D] = LaplacianEigMap(kernelGraph, weightMargin);
    sorted = sort(V(:,2));
    
    zs = zeros(dim(3), dim(4), 'single');
    step = floor(dim(3)/groups) +1;
    for i = 1 : dim(4)
        groupid = floor((i-1) * groups / dim(4));
        xmin = sorted(groupid * step + 1);
        xmax = sorted(min(dim(3), step * (groupid + 1)) );
        zs(:, i) = or(V(:,2) < xmin, V(:,2) > xmax)';
    end
    
    
    


    type = 'single';
    weight = init_weight('xavierimproved',dim(1),dim(2),dim(3),dim(4), type);
    
    net.layers{end+1} = struct(...
       'type', 'custom', ...
       'weights', {{weight, zeros(1, dim(4), type)}}, ...
       'learningRate', lrs, ...
       'stride', stride, ...
       'precious', 0, ...
       'pad', pad, ...
       'nnz', nnz, ...
       'useGPU', useGPU, ...
       'zeroid', logical(zs), ...
       'forward', @sparse_conv_forward,...
       'backward', @sparse_conv_backward);
end
