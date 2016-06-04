function net = cnn_cifar_init(varargin)
% the network configurations
add_conv_custom=@add_sparse_conv; % add_conv_custom=@add_conv;
nnz= [3 4 4 4 64];
global nextLayerId
nextLayerId = 0;
%%
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;

% Yang: learning rate is wrong
lr = [1 2] ;

% Define network CIFAR10-quick
net.layers = {} ;

% Block 1
net = add_conv_custom(net, [5 5 3 32], lr, 1, 2, nnz(1));
net.layers{end}.weights{1}=randn(5, 5, 3, 32, 'single')*0.0001;
net = add_pool(net, 'max', [3 3], 2, [0 1 0 1]);
net.layers{end+1} = struct('type', 'relu') ;

% Block 2
net = add_conv_custom(net, [5 5 32 32], lr, 1, 2, nnz(2));
net.layers{end}.weights{1}=randn(5, 5, 32, 32, 'single')*0.01;
net.layers{end+1} = struct('type', 'relu') ;
net = add_pool(net, 'avg', [3 3], 2, [0 1 0 1]);

% Block 3
net = add_conv_custom(net, [5 5 32 64], lr, 1, 2, nnz(3));
net.layers{end}.weights{1}=randn(5, 5, 32, 64, 'single')*0.01;
net.layers{end+1} = struct('type', 'relu') ;
net = add_pool(net, 'avg', [3 3], 2, [0 1 0 1]);

% Block 4
net = add_conv_custom(net, [4 4 64 64], lr, 1, 0, nnz(4));
net.layers{end}.weights{1}=randn(4, 4, 64, 64, 'single')*0.1;
%net.layers{end+1} = struct('type', 'relu') ;

% Block 5, change the lr here.
net = add_conv_custom(net, [1 1 64 10], lr, 1, 0, nnz(5));
net.layers{end}.weights{1}=randn(1, 1, 64, 10, 'single')*0.1;

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


