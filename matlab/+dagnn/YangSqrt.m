classdef YangSqrt < dagnn.ElementWise

  methods
    function outputs = forward(obj, inputs, params)
      in = inputs{1};
      outputs{1} = sign(in).*sqrt(abs(in));
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      ep=1e-1;
      derInputs{1}=derOutputs{1} .* 0.5 ./ (sqrt(abs(inputs{1})) + ep);
      derParams = {} ;
    end

    function obj = YangSqrt(varargin)
      obj.load(varargin) ;
    end
  end
end
