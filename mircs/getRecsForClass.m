function rec_subset = getRecsForClass(recs,cls)
% cls = catDir
sel_ = false(size(recs));
for k = 1:length(recs)
    clsinds=strmatch(cls,{recs(k).objects(:).class},'exact');
    sel_(k) = any(clsinds);
end
rec_subset = recs(sel_);
end