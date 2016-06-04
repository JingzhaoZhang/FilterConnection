classdef YangSumPool < dagnn.Filter
  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = sum(sum(inputs{1},1),2) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      sz = size(inputs{1});
      derInputs{1} = repmat(derOutputs{1}, [sz(1) sz(2) 1 1]);
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      sz = inputSizes{1};
      sz(1) = 1;
      sz(2) = 1;
      outputSizes{1} = sz;
    end

    function obj = YangSumPool(varargin)
      obj.load(varargin) ;
    end
  end
end
