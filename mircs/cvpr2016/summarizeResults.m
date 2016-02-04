function [summary,sm] = summarizeResults(res,all_feats,train_params)

nClasses = length(train_params.classes);
summary =  zeros(length(res),nClasses+1);
subset_names = {};
for iSubset = 1:length(res)
    curRes = res{iSubset};
    f = find(curRes.featureSubset);
    subset_name = [];
    for q = 1:length(f)
        subset_name = [subset_name,all_feats(f(q)).abbr];
        if q < length(f)
            subset_name = [subset_name,'+'];
        end
    end
    subset_names{iSubset} = subset_name;
    res_test = curRes.res_test;
    for iClass = 1:nClasses
        summary(iSubset,1+iClass)=res_test(iClass).info.ap;
    end
end
summary(:,1)=mean(summary(:,2:end),2);
% imagesc(summary)
sm = array2table(summary);
classNames = train_params.classNames(1:length(train_params.classes));
% classNames = {};
% for iClass = 1:5
%     classNames{iClass} = fra_db(find([fra_db.classID]==iClass,1,'first')).class;
% end
% 
% classNames = {'Drink','Smoke','Blow','Brush','Phone'};
% nchoosek(5,3)+nchoosek(5,2)+5
sm = [table((1:size(sm,1))'), sm]; % add experiment indices
sm.Properties.VariableNames=['exp_id', 'mean', classNames];
sm.Properties.RowNames = subset_names;

%sm
sm = sortrows(sm,'mean');
sm