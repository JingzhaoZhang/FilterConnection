%% default configuration
run ../../matlab/vl_setupnn.m

opts=struct('modelType', 'lenet',...
            'whitenData', false,...
            'contrastNormalization', false,...
            'train', struct('gpus',1));
% some values about:
% learning rate, weightDecay, batchSize, momentum is in cnn_cifar_init
%%
opts.train.gpus=8;
[net, info]=cnn_cifar(opts);
