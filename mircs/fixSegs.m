function segs = fixSegs(segs)
    y1 = segs(:,2);
    y2 = segs(:,4);
    toFix = y2 < y1;
    segs(toFix,:) = segs(toFix,[3 4 1 2]);
end