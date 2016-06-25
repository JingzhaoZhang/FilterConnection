function net = le_cnn_cifar_init(varargin)
% the network configurations
%add_conv_custom=@add_sparse_conv; % add_conv_custom=@add_conv;
add_conv_custom = @(net, dim, lrs, stride, pad, nnz, useGPU) add_le_sparse_conv(net, dim, lrs, stride, pad, nnz, useGPU, 2);
%nnz= [3 4 4 4 64];
%nnz  = [3, 32, 32, 64, 64];
nnz = [3, 16,16,32,64];
global nextLayerId
nextLayerId = 0;
%%
opts.networkType = 'simplenn' ;
opts.gpus=[];
opts = vl_argparse(opts, varargin) ;


tag = 'sparse-nnz_3_32_32_64_64';
expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
%Load the last epoch's result
Epoch = 80;
modelPath = fullfile(expDir, sprintf('net-epoch-%d.mat', Epoch));
netstruct = load(modelPath,'net') ;
originalnet = vl_simplenn_tidy(netstruct.net) ;


% Yang: learning rate is wrong
lr = [1 2] ;

% Define network CIFAR10-quick
net.layers = {} ;
useGPU = numel(opts.gpus) > 0;
% Block 1
net = add_conv(net, [5 5 3 32], lr, 1, 2, nnz(1));
net.layers{end}.weights{1}=gather(originalnet.layers{1}.weights{1});
net = add_pool(net, 'max', [3 3], 2, [0 1 0 1]);
net.layers{end+1} = struct('type', 'relu') ;

% Block 2
net = add_conv_custom(net, [5 5 32 32], lr, 1, 2, nnz(2), useGPU);
net.layers{end}.weights{1}=gather(originalnet.layers{4}.weights{1});
net.layers{end+1} = struct('type', 'relu') ;
net = add_pool(net, 'avg', [3 3], 2, [0 1 0 1]);

% Block 3
net = add_conv_custom(net, [5 5 32 64], lr, 1, 2, nnz(3), useGPU);
net.layers{end}.weights{1}=gather(originalnet.layers{7}.weights{1});
net.layers{end+1} = struct('type', 'relu') ;
net = add_pool(net, 'avg', [3 3], 2, [0 1 0 1]);

% Block 4
net = add_conv_custom(net, [4 4 64 64], lr, 1, 0, nnz(4), useGPU);
net.layers{end}.weights{1}=gather(originalnet.layers{10}.weights{1});
%net.layers{end+1} = struct('type', 'relu') ;

% Block 5, change the lr here.
net = add_conv(net, [1 1 64 10], lr, 1, 0, nnz(5));
net.layers{end}.weights{1}=gather(originalnet.layers{11}.weights{1});

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


