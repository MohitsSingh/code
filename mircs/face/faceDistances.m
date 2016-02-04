function D = faceDistances(xy_1,xy_2)
% Calculates the overall distance between the keypoints of pairs of faces.
% For different numbers of keypoints (e.g, very different poses) the
% distance will be regarded as inf (or a very large number)
D = 10^6*ones(length(xy_1),length(xy_2));

% calculate distances for sets of keypoints with same cardinality
lengths1 = cellfun(@length,xy_1(:));
lengths2 = cellfun(@length,xy_2(:));
L = unique([lengths1;lengths2]);
for iL = 1:length(L) %
    t1 = find(lengths1==L(iL));
    t2 = find(lengths2==L(iL));
    a = cat(2,xy_1{t1});
    b = cat(2,xy_2{t2});
    d = l2(a',b');
    D(t1,t2) = d;
    %     D = reshape(1:25,5,5)
    %     D([1 3],[3 5]) = [100,200;300,400];
    
end


end