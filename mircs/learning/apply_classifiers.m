function res = apply_classifiers(classifiers,features,labels,params,islibsvm,valids,std_data)
classes = params.classes;
nClasses = length(classes);
res = struct('class_id',{},'performance',{});
if nargin < 5 || isempty(islibsvm)
    islibsvm = false;
end
if nargin < 6 || isempty(valids)
    valids = true(1,length(labels));
end

if nargin < 7
    std_data = [];
end
if ~islibsvm
    [ws bs] = get_w_from_classifiers(classifiers);
    
    all_res = -inf(length(classifiers),size(features,2));
    if iscell(features) % multiple features per instance
        for iClassifier = 1:size(ws,2)
            disp(iClassifier)
            w = ws(:,iClassifier);
            b = bs(iClassifier);
            
            www =zeros(size(features,2),3);                      
            for iFeat = 1:size(features,2)
                
                
                % the last set of features may be of multiple values                                
                
                f = features(:,iFeat);
                s =size(f{end},2); 
                if (s>1)
                    for iii = 1:size(f,1)-1
                        f{iii} = repmat(f{iii},1,s);
                    end
                end
                        
                f = cat(1,f{:});
%                 f = f.*(f>0);
                %f = cat(1,features{:,iFeat});                
                
                if ~isempty(std_data)
                    f = standardizeCols(f',std_data.mu,std_data.sigma2);
                    f = f';
                end
                
%                 if iscell(f)
%                     f = f{1};
%                 end
                if (any(f))
                    zz = w'*f;
                    www(iFeat,1:length(zz)) = zz;
                    all_res(iClassifier,iFeat) = max(w'*f,[],2)+b;
%                     all_res(iClassifier,iFeat) = mean(w'*f,2)+b;
                end
            end
        end
    else
        all_res = bsxfun(@plus,ws'*features,bs);
    end
    
else
    all_res = {};
    for t = 1:length(classifiers)
        [~,~,p] = svmpredict(zeros(size(features,2),1),double(features)',classifiers(t).classifier_data);
        all_res{t} = row(p);
        break
    end
    all_res = repmat(all_res,1,5);
    all_res = cat(1,all_res{:});
end

for iClass = 1:nClasses
    curClass = classes(iClass);
    res(iClass).class_id = curClass;
    curScores = all_res(iClass,:);
    if ~(strcmp(params.task,'regression'))
%         curLabels = labels(:,iClass);
%     else
        if min(size(labels))==1
            curLabels = (labels==curClass)*2-1;    
        else
            curLabels = 2*(labels(:,classes(iClass))==1)-1;            
        end
        curScores(isnan(curScores)) = -inf;        
        [res(iClass).recall, res(iClass).precision, res(iClass).info] = vl_pr(curLabels,curScores);
    end
    res(iClass).curScores = curScores;
    %     curClass
    %     class_names{curClass}
    %     clf; vl_pr(curLabels,curScores); pause
end

