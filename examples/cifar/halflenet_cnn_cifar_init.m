function net = doublelenet_cnn_cifar_init(varargin)
% the network configurations
%add_conv_custom=@add_sparse_conv; % add_conv_custom=@add_conv;
%nnz= [3 4 4 4 64];
%nnz  = [3, 32, 32, 64, 64];
global nextLayerId
nextLayerId = 0;
%%
opts.networkType = 'simplenn' ;
opts.gpus=[];
opts.nnz = [3, 16,16,32,64];
opts.add_conv_custom = @add_sparse_conv;
opts.weightinit = 'xavierimproved';
opts = vl_argparse(opts, varargin) ;

add_conv_custom = opts.add_conv_custom;
nnz = opts.nnz;
% Yang: learning rate is wrong
lr = [1 2] ;

% Define network CIFAR10-quick
net.layers = {} ;
useGPU = numel(opts.gpus) > 0;
% Block 1
net = add_conv(net, [5 5 3 64], lr, 1, 2, nnz(1));
%net.layers{end}.weights{1}=randn(5, 5, 3, 64, 'single')*0.0001;
net.layers{end}.weights{1} = init_weight(opts.weightinit, 5, 5, 3, 64, 3, nnz(1), 'single');
net = add_pool(net, 'max', [3 3], 2, [0 1 0 1]);
net.layers{end+1} = struct('type', 'relu') ;

% Block 2
net = add_conv_custom(net, [5 5 64 64], lr, 1, 2, nnz(2), useGPU);
%net.layers{end}.weights{1}=randn(5, 5, 64, 64, 'single')*0.01;
net.layers{end}.weights{1} = init_weight(opts.weightinit, 5, 5, 64, 64, nnz(1), nnz(2), 'single');
net.layers{end+1} = struct('type', 'relu') ;
net = add_pool(net, 'avg', [3 3], 2, [0 1 0 1]);

% Block 3
net = add_conv_custom(net, [5 5 64 128], lr, 1, 2, nnz(3), useGPU);
%net.layers{end}.weights{1}=randn(5, 5, 64, 128, 'single')*0.01;
net.layers{end}.weights{1} = init_weight(opts.weightinit, 5, 5, 64, 128, nnz(2), nnz(3), 'single');
net.layers{end+1} = struct('type', 'relu') ;
net = add_pool(net, 'avg', [3 3], 2, [0 1 0 1]);

% Block 4
net = add_conv_custom(net, [4 4 128 128], lr, 1, 0, nnz(4), useGPU);
%net.layers{end}.weights{1}=randn(4, 4, 128, 128, 'single')*0.1;
net.layers{end}.weights{1} = init_weight(opts.weightinit, 4, 4, 128, 128, nnz(3), nnz(4), 'single');

%net.layers{end+1} = struct('type', 'relu') ;

% Block 5, change the lr here.
net = add_conv(net, [1 1 128 10], lr, 1, 0, nnz(5));
net.layers{end}.weights{1} = init_weight(opts.weightinit, 1, 1, 128, 10, nnz(4), nnz(5), 'single');

%net.layers{end}.weights{1}=randn(1, 1, 128, 10, 'single')*0.1;

% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

% Meta parameters
net.meta.inputSize = [32 32 3] ;
net.meta.trainOpts.learningRate = [0.001*ones(1,90) 0.0001*ones(1,30)] ;
net.meta.trainOpts.weightDecay = 0.004 ;
net.meta.trainOpts.batchSize = 100 ; 
net.meta.trainOpts.momentum = 0.9;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
             {'prediction','label'}, 'error') ;
  otherwise
    assert(false) ;
end


% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, effective_in, effective_out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.
    switch lower(opts)
      case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
      case 'xavier'
        sc = sqrt(3/(h*w*effective_in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
      case 'xavierimproved'
        sc = sqrt(2/(h*w*effective_out)) ;
        weights = randn(h, w, in, out, type)*sc ;
      otherwise
        error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
    end
end

