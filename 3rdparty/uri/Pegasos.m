classdef Pegasos
    
    properties
        w
        optLambda
        optAvgPrec
    end
    
    methods (Hidden = false)
        function obj = Pegasos(X, y, varargin)
            y = 2*single(y > 0)-1;
            %             X = single(X);
            
            
            
            lambda = checkVarargin(varargin, 'lambda', logspace(-4,-1,3));
            forceCrossVal = checkVarargin(varargin, 'forceCrossVal ', true);
            bias = checkVarargin(varargin, 'bias ', 1);
            
            %             lambda = 0.0001;
            %             lambda =  .1; %good
            %             lambda = checkVarargin(varargin, 'lambda', [.1 .01 .001 .0001]);
            %             balanceDatasets = checkVarargin(varargin, 'balanceDatasets', false);
            %                         lambda = checkVarargin(varargin, 'lambda',linspace(5e-3,1e-1,10));
            
            foldNum = checkVarargin(varargin, 'foldNum', 5);
            %                         foldNum = checkVarargin(varargin, 'foldNum', 10);
            %             lambda = checkVarargin(varargin, 'lambda', 1e-4);
            avgPrec = 0;
            maxAvgPrec = 0;
            n = length(lambda);
            if (n > 1 || forceCrossVal)
                cvIdx = crossvalind('Kfold', length(y), foldNum);
                maxAvgPrec = -inf;
                perm = randperm(n);
                maxModel = [];
                maxLambda = lambda(1);
                for ii = 1:n
                    
                    i = ii;
%                     i
                    %                     i = perm(ii);
                    avgPrec = zeros(1, foldNum);
                    for j = 1:foldNum
                        
                        
                        
                        
                        trIdx = cvIdx ~= j;
                        teIdx = ~trIdx;
                        %                         X_tr = single(full());
                        %wi = Pegasos.train(single(full(X(:,trIdx))), y(trIdx), lambda(i));
                        wi = Pegasos.train((full(X(:,trIdx))), y(trIdx), lambda(i),'bias',bias);
                        %yhat = obj.test(single(full(X(:,teIdx))), wi);
                        yhat = obj.test((full(X(:,teIdx))), wi);
                        avgPrec(j) = Pegasos.averagePrecision(y(teIdx), yhat);
                        %if (mod(j,10)==0)
                            disp(['foldNum : ', num2str(j) '/' num2str(foldNum),...
                                ' lambda: ' , num2str(lambda(i)),...
                                'avg. prec: ', num2str(avgPrec(j))]);
                        %end
                    end
                    meanAvgPrec = mean(avgPrec);
                    if (meanAvgPrec > maxAvgPrec)
                        maxAvgPrec = meanAvgPrec;
                        maxLambda = lambda(i);
                        maxModel = wi;
                    end
                end
                obj.optAvgPrec = maxAvgPrec;
                obj.optLambda = maxLambda;
            else
                maxLambda = lambda;
            end
            
            disp(['Finished optimizing lambda. lambda = ' num2str(maxLambda) ...
                ' , best mean avg prec = ' num2str(obj.optAvgPrec)]);
            
            
            retrain = true;
            if (retrain)
                
                % training
%                 maxAvgPrec = -inf;
                %             for j = 1:foldNum
                %                 X = single(full(X));
                X = (full(X));
                wi = Pegasos.train(X, y, maxLambda,'bias',bias);
                maxModel = wi;
%                 yhat = obj.test(X, wi);
%                 avgPrec = Pegasos.averagePrecision(y, yhat);
%                 if (avgPrec > maxAvgPrec)
%                     maxAvgPrec = avgPrec;
%                     maxModel = wi;
%                 end
                
            end
            disp(['finished training :',...
                ' avg. prec: ', num2str(avgPrec)]);
            
            %             end
            obj.w = maxModel;
            obj.optLambda = maxLambda;
            obj.optAvgPrec = maxAvgPrec;
        end
        
        function [Yhat f] = test(obj, X, w)
            
            if (nargin < 3)
                w = obj.w;
            end
            
            %f = (w'*[full(X); ones(1, size(X,2), 'single')])';
            f = (w'*[full(X); ones(1, size(X,2))])';
            Yhat = sign(f);
        end
    end
    
    methods (Hidden = true, Static = true)
        
        function w = train(X, y, lambda, varargin)
            bias = checkVarargin(varargin, 'bias', 0);
            %             iterNum = checkVarargin(varargin, 'iterNum', 1000);
            [w b info] = vl_svmtrain(X, double(y), lambda, 'BiasMultiplier', bias);%, 'MaxNumIterations', iterNum);
            edit vl_svmtrain
            %             [w b info] = vl_svmtrain(X, double(y), lambda, 'BiasMultiplier', bias, 'MaxNumIterations', iterNum);
            %             [w b info] = vl_pegasos(X, int8(y), lambda, 'BiasMultiplier', bias);%, 'MaxIterations', iterNum);
            %'HOMKERMAP',1);
            
            w = [w;b];
        end
        
        function avgPrec = averagePrecision(y, yhat)
            
            confMat = accumarray(round(.5*[y yhat]+1.5), ones(size(y)), [2 2]);
            %             disp(confMat)
            prec = diag(bsxfun(@rdivide, confMat, sum(confMat,2)));
            avgPrec = mean(prec);
            %             mean(prec(prec > 0));
        end
    end
end