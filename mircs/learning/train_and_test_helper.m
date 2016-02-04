function res = train_and_test_helper(all_feats,labels,isTrain,valids,train_params)

if nargin < 4 || isempty(valids)
    valids = true(size(labels,1),1);
end

%res = struct('participating_features',{},'class',{},'performance',{});
subsets = allSubsets(length(all_feats));
subsets = subsets(2:end,:);
% subsets
m = sum(subsets,2);
% subsets(m>1,:) = [];
subsets(m < train_params.minGroupSize | m > train_params.maxGroupSize,:) = [];
% 7 ,:) = []; % don't allow combinations of more than 4
% subsets=true(1,8);
% subsets = subsets(82,:);
% m(m>4,:) = [];
% subsets(m<3,:) = []; % or less than 3
% subsets(m==1,:) = [];
res = {};

for t = 1:length(all_feats)
    
end

for iSubset = 1:size(subsets,1)
    iSubset/size(subsets,1)
    curSubset = subsets(iSubset,:);
    
    % transform the feature
    
    feats = {};
    for t = 1:length(curSubset)
        if curSubset(t)
            % make sure all features are in cell array form - but do it only if the
            % features are required.
            
            if ~iscell(all_feats(t).feats)
                all_feats(t).feats = mat2cell2(all_feats(t).feats,[1,size(all_feats(t).feats,2)]);
            end
            feats{t} = all_feats(t).feats;
        end
    end
    feats = cat(1,feats{:});
    train_features = feats(:,isTrain);
    if (iscell(train_features))
        r = cell(1,size(train_features,2));
        for q = 1:size(r,2)
            r{q} = cat(1,train_features{:,q});
        end
        train_features = cat(2,r{:});
        %          train_features = cat(2,train_features{:}');
        %          if (iscell(train_features))
        %             train_features = cell2mat(train_features);
        %         end
    end
    train_labels = labels(isTrain,:);
    
    
    % do something crazy now
%     'warning!!!!!!!!!!'
    sel_test = ~isTrain;
    
    test_features = feats(:,sel_test);    
    % do standartization if needed
    if train_params.standardize
        [train_features,mu,sigma2] = standardizeCols(train_features');
        train_features = train_features';
        std_data=struct('mu', mu, 'sigma2', sigma2);
    else
        std_data = [];
    end
    
    
    
    test_labels = labels(sel_test,:);        
    %     for iClass = 1:length(train_params)
    
    % get feature subset
    disp('training...')
    res_train = train_classifiers(train_features,train_labels,[],train_params,valids(isTrain));
    test_valids = valids(sel_test);
    disp('testing...')
    res_test = apply_classifiers(res_train,test_features(:,test_valids),test_labels(test_valids,:),train_params,[],[],std_data);
    curResults.featureSubset = curSubset;
    curResults.res_train = res_train;
    curResults.res_test = res_test;
    res{iSubset} = curResults;
    %     end
end

