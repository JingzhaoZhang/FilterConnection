function pre=sparse_conv_backward(layer, pre, now)
    if layer.useGPU
        w1 = gpuArray(layer.weights{1});
        w2 = gpuArray(layer.weights{2});
    else
        w1 = layer.weights{1};
        w2 = layer.weights{2};
    end
    [pre.dzdx, dzdw1, dzdw2] = ...
      vl_nnconv(pre.x,...
      w1, w2,...
      now.dzdx, ...
      'stride', layer.stride, ...
      'pad', layer.pad);
    % clear the zero terms in dzdw1
    dzdw1 = setzero(dzdw1, layer.zeroid);
    
    pre.dzdw={dzdw1, dzdw2};
end

function w=setzero(w, zid)
    [szh, szw, cin, cout] = size(w);
    w = reshape(w, szh, szw, cin*cout);
    w(:,:,zid(:)) = 0;
    w = reshape(w, szh, szw, cin, cout);
end