function [M] = knn_match(x1,x2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    x1 = single(x1)';
    x2 = single(x2)';
    tree = vl_kdtreebuild(x2);
    [ids,dists] = vl_kdtreequery(tree,x2,x1,'MaxNumComparisons',1000);

    %d = l2(double(x1'),double(x2'));
%     [~,min_i] = min(d');min_i = min_i';
%     M1 = [col(1:length(min_i)) min_i]';

     M = [col(1:size(x1,2)) ids(:)]';
end

