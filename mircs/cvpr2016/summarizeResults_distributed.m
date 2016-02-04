function [summary,sm] = summarizeResults_distributed(res,all_feats,train_params)

nClasses = length(train_params.classes);
nFeats = length(all_feats);
summary =  zeros(length(res),length(all_feats)+nClasses+1);
% subset_names = {};
for iSubset = 1:length(res)
    curRes = res{iSubset};
    f = curRes.featureSubset;
    summary(iSubset,1:nFeats) = f;
    res_test = curRes.res_test;
    for iClass = 1:nClasses
        summary(iSubset,1+iClass+nFeats)=res_test(iClass).info.ap;
    end
end
summary(:,nFeats+1)=mean(summary(:,nFeats+2:end),2);
% imagesc(summary)
sm = array2table(summary);
classNames = train_params.classNames;
% classNames = {};
% for iClass = 1:5
%     classNames{iClass} = fra_db(find([fra_db.classID]==iClass,1,'first')).class;
% end
%
% classNames = {'Drink','Smoke','Blow','Brush','Phone'};
% nchoosek(5,3)+nchoosek(5,2)+5
sm = [table((1:size(sm,1))'), sm]; % add experiment indices
sm.Properties.VariableNames=['exp_id',{all_feats.abbr}, 'mean', classNames(1:length(train_params.classes))];

% for t = 1:length(all_feats)
%     curValues = sm{:,t+1};
%     v = {};
%     for q = 1:length(curValues)
%         if curValues(q)
%             v{q} = '+'
%         else
%             v{q} = ' '
%         end
%     end
%     sm{:,t+1} = v';
% end

%sm
sm = sortrows(sm,'mean');
% sm