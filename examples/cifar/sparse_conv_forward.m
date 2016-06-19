% the sparse conv layer should have the same fileds as the normal conv
% layers, that is:
% weights = {filters, bias}
% stride
% pad
% nnz: a new field indicates how many filters are non zero
% zeroid: a cache of non zero indexes
% id: this layer id
function now=sparse_conv_forward(layer, pre, now)
    % set the corresponding filters to 0
    layer.weights{1} = setzero(layer.weights{1}, layer.zeroid);
    % then normal convolution
    if layer.useGPU
        w1 = gpuArray(layer.weights{1});
        w2 = gpuArray(layer.weights{2});
    else
        w1 = layer.weights{1};
        w2 = layer.weights{2};
    end
    now.x = vl_nnconv(pre.x, ...
        w1, w2, ...
        'stride', layer.stride, ...
        'pad', layer.pad);
end

function w=setzero(w, zid)
    [szh, szw, cin, cout] = size(w);
    w = reshape(w, szh, szw, cin*cout);
    w(:,:,zid(:)) = 0;
    w = reshape(w, szh, szw, cin, cout);
end
