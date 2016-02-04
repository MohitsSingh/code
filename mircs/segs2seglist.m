function seglist = segs2seglist(segs)
    seglist = {};
    for k = 1:size(segs,1)
        seglist{k} = [segs(k,1:2);segs(k,3:4)];
    end
end