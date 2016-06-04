function net=add_sparse_conv(net, dim, lrs, stride, pad, nnz)
    % generate the zero locations
    % fix seed in order to be predictable
    prevSeed = rng;
    global nextLayerId
    if isempty(nextLayerId)
        nextLayerId = 1;
    else 
        nextLayerId = nextLayerId + 1;
    end
    id=nextLayerId;
    rng(id);
    zs = zeros(dim(3), dim(4), 'like', pre.x);
    for i = 1 : cout
        zs(randsample(cin, cin-layer.nnz), i)=1;
    end
    layer.zeroid=logical(zs);

    rng(prevSeed);
    

    type = 'single';
    weight = init_weight('xavierimproved',dim(1),dim(2),dim(3),dim(4), type);
    
    net.layers{end+1} = struct(...
       'type', 'custom', ...
       'weights', {{weight, zeros(1, dim(4), type)}}, ...
       'learningRate', lrs, ...
       'stride', stride, ...
       'pad', pad, ...
       'nnz', nnz, ...
       'id', id, ...
       'forward', @sparse_conv_forward,...
       'backward', @sparse_conv_backward);
end
