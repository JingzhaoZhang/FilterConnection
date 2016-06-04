classdef YangL2norm < dagnn.ElementWise

  methods
    function outputs = forward(obj, inputs, params)
      now = yang_l2norm_forward([], struct('x', inputs{1}), []);
      outputs{1} = now.x;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      pre = yang_l2norm_backward([], struct('x', inputs{1}), struct('dzdx', derOutputs{1}));
      derInputs = {pre.dzdx};
      derParams = {} ;
    end

    function obj = YangL2norm(varargin)
      obj.load(varargin) ;
    end
  end
end
