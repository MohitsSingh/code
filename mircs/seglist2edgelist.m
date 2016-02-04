function edgelist = seglist2edgelist(seglist)
edgelist = {};
for k = 1:length(seglist)
    L = seglist{k};
    edgelist{k} = [L(:,1:2);L(end,3:4)];
end
end