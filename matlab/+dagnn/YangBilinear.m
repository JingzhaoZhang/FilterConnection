classdef YangBilinear < dagnn.Filter

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnbilinearpool(inputs{1});
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnbilinearpool(inputs{1}, derOutputs{1});
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = [1 1] ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      sz = inputSizes{1};
      sz(1) = 1;
      sz(2) = 1;
      sz(3) = sz(3)*sz(3);
      outputSizes{1} = sz;
    end

    function obj = YangBilinear(varargin)
      obj.load(varargin) ;
    end
  end
end
