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
opts.train.learningRate = [0.001 * ones(250, 1) ];
opts.train.numEpochs = numel(opts.train.learningRate);
opts.train.continue = false;
opts.modelType = 'lagragian';
tag = 'sparse-le-epoch80';

expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);
%%
opts.train.gpus=7;
opts.train.learningRate = [0.001 * ones(250, 1) ];
opts.train.numEpochs = numel(opts.train.learningRate);
tag = 'sparse-nnz_3_16_16_32_64';

expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);
%%
opts.train.gpus=8;
tag = 'sparse-nnz_3_32_32_64_64';
opts.train.learningRate = [0.001 * ones(80, 1) ];
opts.train.numEpochs = numel(opts.train.learningRate);
expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);
%%
opts.train.gpus=6;
tag = 'doublelenet-sparse-nnz_3_64_64_128_128';
opts.nnz = [3, 64, 64, 128, 128];
opts.add_conv_custom = @add_sparse_conv;
opts.train.learningRate = [0.001 * ones(150, 1) ];
opts.train.numEpochs = numel(opts.train.learningRate);
expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);
%%
opts.train.gpus=7;
tag = 'doublelenet-sparse-nnz_3_32_32_64_64';
opts.nnz = [3, 32, 32, 64,64];
opts.add_conv_custom = @add_sparse_conv;
opts.train.learningRate = [0.001 * ones(150, 1) ];
opts.train.numEpochs = numel(opts.train.learningRate);
expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);


%%
opts.train.gpus=6;
tag = 'finetune-doublelenet-nnz_3_64_64_128_128';
opts.nnz = [3, 64, 64, 128, 128];
opts.modelType = 'double_lenet';
opts.add_conv_custom = @add_sparse_conv;
opts.train.learningRate = [1e-4 * ones(1,150), 1e-5 * ones(1, 100)];
opts.train.numEpochs = numel(opts.train.learningRate);
expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);


%%
opts.train.gpus=5;
tag = 'finetune-doublelenet-nnz_3_32_32_64_64';
opts.nnz = [3, 32, 32, 64, 64];
opts.modelType = 'double_lenet';
opts.add_conv_custom = @add_sparse_conv;
opts.train.learningRate =  [1e-4 * ones(1,200), 1e-5 * ones(1, 100)];
opts.train.numEpochs = numel(opts.train.learningRate);
expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);
%%
opts.train.gpus=5;
tag = 'deeplenet-nnz_3_32_32_64_64';
opts.nnz = [3, 32, 32, 32, 32, 64, 64, 64, 64];
opts.modelType = 'deep_lenet';
opts.add_conv_custom = @add_sparse_conv;
opts.train.learningRate =  [1e-2 * ones(1,50), 1e-3 * ones(1,100), 1e-4 * ones(1, 50)];
opts.train.numEpochs = numel(opts.train.learningRate);
expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);

%%
opts.train.gpus=5;
tag = 'finetune-lenet-nnz_3_32_32_64_64';
opts.nnz  = [3, 32, 32, 64, 64];
opts.modelType = 'lenet';
opts.add_conv_custom = @add_sparse_conv;
opts.train.learningRate =  [1e-3 * ones(1,100), 0.5e-3 * ones(1,100), 0.2e-3*ones(1,100)];
opts.train.numEpochs = numel(opts.train.learningRate);
expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);
%%
opts.train.gpus=6;
tag = 'narrow_deeplenet-nnz-16_16_16_16_32_32_64_64';
opts.nnz = [3, 16, 16, 16, 16, 32, 32, 64, 64];
opts.modelType = 'narrow_deep_lenet';
opts.add_conv_custom = @add_sparse_conv;
opts.train.learningRate =  [ 1e-4 * ones(1, 50), 0.5e-4 * ones(1,100), 0.2e-4 * ones(1, 50) 1e-5 * ones(1,100)];
opts.train.numEpochs = numel(opts.train.learningRate);
expDir = fullfile(vl_rootnn, 'data', ['cifar-' tag]);
cnn_cifar(opts, 'expDir', expDir);


