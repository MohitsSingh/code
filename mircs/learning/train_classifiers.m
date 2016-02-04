function res = train_classifiers( train_data,train_labels,train_values,train_params,valids)
% res = train_classifiers( train_data,train_labels,train_params) convenience wrapper
% function to train classifiers for several classes, with cross-validation,
% data balancing, etc.
if nargin < 5 || isempty(valids)
    valids = true(size(train_data,2),1);
end

% train_data = train_data.*(train_data>0);

classes = train_params.classes;
nClasses = length(classes);
toBalance = train_params.toBalance;
lambdas = train_params.lambdas;
if isfield(train_params,'min_pos_ovp')
    train_labels(train_values < train_params.min_pos_ovp) = -1;
end

if isfield(train_params,'max_neg_ovp')
    toIgnore = train_values < train_params.min_pos_ovp & train_values >= train_params.max_neg_ovp;
    train_data = train_data(:,~toIgnore);
    train_labels = train_labels(~toIgnore);
    train_values = train_values(~toIgnore);
    valids = valids(~toIgnore);
end

res = struct('class_id',{},'classifier_data',{});
% train 1 vs all classifiers for all classes
for iClass = 1:nClasses
    res(iClass).class_id = classes(iClass);
    iClass
    if strcmp(train_params.task, 'classification')
        
        if all(size(train_labels) == size(valids))
            curLabels = 2*col(train_labels==classes(iClass))-1;
        else
            curLabels = 2*(train_labels(:,classes(iClass))==1)-1;
        end
        curValids = reshape(valids,size(curLabels));        
        curTrainData = train_data;
       [curTrainData,curLabels,curValids] = balanceData(train_data,curLabels,toBalance,curValids);
%       warning('ignoring balancing for now')
      
      
%        [curTrainData,curLabels,curValids] = deal(train_data,train_labels(:,
        
        if train_params.hardnegative
            classifier_data = hardNegativeMining(curTrainData,curLabels);
            res(iClass).classifier_data = classifier_data;
        else            
            if length(lambdas)==1
                [w b info] = vl_svmtrain(curTrainData(:,curValids), double(curLabels(curValids)), lambdas, 'BiasMultiplier', 1);%, 'MaxNumIterations', iterNum);
%                 optsstring=sprintf('-s 0 -c %f -w1 5 -w2 1',lambdas);
%                 curmodel = train(double(curLabels(curValids)),sparse(double(curTrainData(:,curValids))),optsstring,'col');
%                 w = curmodel.w(:);b = curmodel.bias;
                classifier_data=struct('w',[w;b]);
            else
                classifier_data = Pegasos(curTrainData(:,curValids),curLabels(curValids),...
                    'lambda', lambdas,'forceCrossVal',true);
            end
            res(iClass).classifier_data = classifier_data;
        end
        %         res(iClass).classifier_data = struct('w',classifier_data.w,'optLambda',classifier_data.optLambda,...
        %             'optAvgPrec',classifier_data.optAvgPrec);
    elseif strcmp(train_params.task, 'classification_rbf')                        
        [curTrainData,curLabels]=balanceData(train_data,2*col(train_labels==classes(iClass))-1,toBalance);
        classifier_data = svmtrain(double(curLabels),(double(curTrainData)'), '-s 0 -t 2');
        %res(iClass).classifier_data = struct('w',classifier_data.w,'optLambda',classifier_data.optLambda,...
        %   'optAvgPrec',classifier_data.optAvgPrec);
        res(iClass).classifier_data = classifier_data;
    elseif strcmp(train_params.task, 'regression')
        trainOpts = sprintf('-s 11 -B 1 -e %f',lambdas);
        %curTrainLabels = train_values.*(train_labels==classes(iClass));
        %curTrainLabels = train_labels(:,iClass);
        curTrainLabels = train_values;%(:,iClass);
        %         curTrainLabels = train_values;
        regressionModel = train(double(curTrainLabels),sparse(double(train_data)), trainOpts,'col');
        res(iClass).classifier_data.w = regressionModel.w(:);
    end
end
end
