function segs = seglist2segs(seglist)
segs = {};
for k = 1:length(seglist)
    c = seglist{k};
    segs{k} = [c(1:end-1,:),c(2:end,:)];
end

segs = cat(1,segs{:});

end