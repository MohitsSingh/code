function recs = readAllRecs(VOCopts,ids)
recs = cell(1,length(ids));
for k = 1:length(ids)
    k
    recs{k} = PASreadrecord(sprintf(VOCopts.annopath,ids{k}));
end
end