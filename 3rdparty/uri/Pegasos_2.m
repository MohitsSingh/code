classdef Pegasos
    
    properties
        w
        optLambda
        optAvgPrec
    end
    
    methods (Hidden = false)
        function obj = Pegasos(X, y, varargin)
            y = 2*single(y > 0)-1;
            %X = single(X);
            lambda = checkVarargin(varargin, 'lambda', 5e-7:5e-6:1e-4);
            foldNum = checkVarargin(varargin, 'foldNum', 5);

            n = length(lambda);           
            if (n > 1)
                cvIdx = crossvalind('Kfold', length(y), foldNum);
                maxAvgPrec = -inf;
                perm = randperm(n);
                for ii = 1:n
                    i = perm(ii);
                    avgPrec = zeros(1, foldNum);
                    for j = 1:foldNum
                                                
                        trIdx = cvIdx ~= j;
                        teIdx = ~trIdx;
                        psix = hkm(single(full(X(trIdx, :)))');
                        wi = Pegasos.train(psix, y(trIdx), lambda(i));                            
                        yhat = obj.test(psix', wi);
                        avgPrec(j) = Pegasos.averagePrecision(y(teIdx), yhat);
                        disp(['foldNum : ', num2str(j) '/' num2str(foldNum),...
                            ' lambda: ' , num2str(lambda(i)),...
                            'avg. prec: ', num2str(avgPrec(j))]);
                    end
                    meanAvgPrec = mean(avgPrec);
                    if (meanAvgPrec > maxAvgPrec)
                        maxAvgPrec = meanAvgPrec;
                        maxLambda = lambda(i);
                    end                    
                end
                obj.optAvgPrec = maxAvgPrec;
                obj.optLambda = maxLambda;                
            else
                maxLambda = lambda;
            end
            
            % training
            maxAvgPrec = -inf;
            for j = 1:foldNum                
                wi = Pegasos.train(X, y, maxLambda);
                yhat = obj.test(X, wi);
                avgPrec = Pegasos.averagePrecision(y, yhat);
                if (avgPrec > maxAvgPrec)
                    maxAvgPrec = avgPrec;
                    maxModel = wi;
                end                            
            end
            obj.w = maxModel;
        end
        
        function [Yhat f] = test(obj, X, w)
            
            if (nargin < 3)
                w = obj.w;
            end
                       
            f = [single(X) ones(size(X,1), 1, 'single')]*w;
            Yhat = sign(f);
        end
    end
    
    methods (Hidden = true, Static = true)
        
        function w = train(X, y, lambda, varargin)
                        
            bias = checkVarargin(varargin, 'bias', 1);
            iterNum = checkVarargin(varargin, 'iterNum', 1e3);            
            w = vl_pegasos(X, int8(y)', lambda, 'BiasMultiplier', bias, 'NumIterations', iterNum);
        end
        
        function avgPrec = averagePrecision(y, yhat)
            confMat = accumarray(.5*[y yhat]+1.5, ones(size(y)), [2 2]);
            prec = diag(bsxfun(@rdivide, confMat, max(sum(confMat,1),1)));
            avgPrec = mean(prec(prec > 0));
        end        
        
    end
end