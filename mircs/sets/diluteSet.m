function [new_ids,new_labels] = diluteSet(ids,labels,k1,k2)
%DILUTESET dilutes the set of samples by returning each k1'th true samples
% and t2'th false sample where k1 and k2 are given. 
%   Detailed explanation goes here
    if (nargin < 4)
        k2 = k1;
    end
    
    
    ids_t = ids(labels);
    ids_t = ids_t(1:k1:end);
    ids_f = ids(~labels);
    ids_f = ids_f(1:k2:end);
    new_ids = [ids_t(:);ids_f(:)];    
    labels_t = labels(labels);
    labels_f = labels(~labels);
    labels_t = labels_t(1:k1:end);
    labels_f = labels_f(1:k2:end);
    new_labels = [labels_t(:);labels_f(:)];
    
end

