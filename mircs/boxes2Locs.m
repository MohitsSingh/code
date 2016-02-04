function locs = boxes2Locs(conf,boxes,ids)
lengths = cellfun(@(x) size(x,1),boxes);
locs = zeros(sum(lengths),12);
c = 0;
for k = 1:length(boxes)
    k
    if (lengths(k))
        [~,xmin,ymin] = toImage(conf,getImage(conf,ids{k}),0,1);
        boxes{k}(:,[1 3]) = boxes{k}(:,[1 3])-xmin;
        boxes{k}(:,[2 4]) = boxes{k}(:,[2 4])-ymin;
        locs(c+1:c+lengths(k),1:4) = boxes{k};
        locs(c+1:c+lengths(k),11) = k;
        c = c+lengths(k);
    end
end
end