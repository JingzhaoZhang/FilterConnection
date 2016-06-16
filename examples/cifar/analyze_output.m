clear all, close all, clc;
%% default configuration
run ../../matlab/vl_setupnn.m

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
tag = 'fully_connected';

imdb = load(opts.imdbPath);
opts.expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);

%Load the last epoch's result
Epoch = 120;
modelPath = fullfile(opts.expDir, sprintf('net-epoch-%d.mat', Epoch));
load(modelPath,'net', 'stats') ;
net = vl_simplenn_tidy(net) ;
net = vl_simplenn_move(net, 'gpu') ;
im = gpuArray(imdb.images.data(:,:,:,10001:10100));
res = vl_simplenn(net, im);


%We can do some analysis on the outputs of relu units
layer_idx = 12;
res_local = gather(res(layer_idx).x);

pairwiseMetric = @covKernel;
weightMargin = 0;
kernelGraph = pairwiseMetric(res_local);
[V,D] = LaplacianEigMap(kernelGraph, weightMargin);


figure(1);
hold on;
plot(V(:,2),V(:,3),'o');
kernelNum = size(kernelGraph);
kernelNum = kernelNum(1);
for i=1:kernelNum
    text(V(i,2),V(i,3),num2str(i));
end
hold off;