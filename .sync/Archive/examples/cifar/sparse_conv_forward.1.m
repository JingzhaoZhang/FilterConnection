% the sparse conv layer should have the same fileds as the normal conv
% layers, that is:
% weights = {filters, bias}
% stride
% pad
% nnz: a new field indicates how many filters are non zero
% zeroid: a cache of non zero indexes
% id: this layer id
function now=sparse_conv_forward(layer, pre, now)
    % generate the non zero locations
    if ~isfield(layer, 'zeroid') || ...
       (isa(layer.zeroid, 'gpuArray') && ~existsOnGPU(layer.zeroid))
        % generate the zero locations
        % fix seed in order to be predictable
        prevSeed = rng;
        global nextLayerId
        if isempty(nextLayerId)
            nextLayerId = 1;
        else 
            nextLayerId = nextLayerId + 1;
        end
        layer.id=nextLayerId;
        rng(layer.id);
        % assume the filter size: h*w*cin*cout
        [h, w, cin, cout] = size(layer.weights{1});
        zs = zeros(cin, cout, 'like', pre.x);
        for i = 1 : cout
            zs(randsample(cin, cin-layer.nnz), i)=1;
        end
        layer.zeroid=logical(zs);
        % set the corresponding filters to 0
        layer.weights{1} = setzero(layer.weights{1}, layer.zeroid);
        
        rng(prevSeed);
    end
    
    % then normal convolution
    now.x = vl_nnconv(pre.x, ...
        layer.weights{1}, layer.weights{2}, ...
        'stride', layer.stride, ...
        'pad', layer.pad);
end

function w=setzero(w, zid)
    [szh, szw, cin, cout] = size(w);
    w = reshape(w, szh, szw, cin*cout);
    w(:,:,zid(:)) = 0;
    w = reshape(w, szh, szw, cin, cout);
end
