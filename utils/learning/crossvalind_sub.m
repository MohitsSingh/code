function inds_folds = crossvalind_sub(inds,k)
%inds_folds = CROSSVALIND_SUB(inds,k) same as crossvalind but makes sure
%that if inds(i) == inds(j) then i and j have the same cross-validation
%index. Assumes that the number of appearances of each element of inds is
%roughly the same.
%   Detailed explanation goes here
u = unique(inds);
u_inds = crossvalind('Kfold',length(u),k);
inds_folds = zeros(size(inds));
for t = 1:k
    cur_group = u(u_inds==t);
    inds_folds(ismember(inds,cur_group)) = t;
end

end

