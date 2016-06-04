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
opts.train.gpus=8;
tag = 'sparse-lr_200_10';

expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);
%%
opts.train.gpus=8;
tag = 'sparse-nnz_32_64';

expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);
%%
opts.train.gpus=7;
tag = 'sparse-nnz_3_32_16_32_64';

expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);
%%
opts.train.gpus=7;
tag = 'sparse-nnz_3_16_16_32_64';

expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);
%%
opts.train.gpus=8;
tag = 'sparse-nnz_3_4_4_4_64v3';

expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);