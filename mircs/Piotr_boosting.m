classdef Piotr_boosting
    
    properties
        model
    end
    
    methods (Hidden = false)
        function obj = Piotr_boosting(X, y, varargin)
            pBoost=struct('nWeak',1024,'verbose',1,'pTree',struct('maxDepth',2,'nThreads',1),'discrete',1);
            obj.model = adaBoostTrain(X(:,y==-1)',X(:,y==1)',pBoost);            
        end
        function [Yhat f] = test(obj, X)
                f = adaBoostApply(X',obj.model,[],[],1);
                Yhat = f>0;
            end
    end
end