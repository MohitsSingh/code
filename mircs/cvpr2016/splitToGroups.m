function [values,inds] = splitToGroups(v,all_inds)
inds_u = unique(all_inds);
values = {};
inds = {};
for iInd = 1:length(inds_u)
    sel_ = find(all_inds==inds_u(iInd));
    values{iInd} = v(sel_);
    inds{iInd} = sel_;
end


