function [ws bs]= get_w_from_classifiers(classifiers)
datas = [classifiers.classifier_data];
ws = cat(2,datas.w);
bs = ws(end,:)';
ws = ws(1:end-1,:);