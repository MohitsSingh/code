function s = ucmScore(ucm,xy)
    t = inImageBounds(size(ucm),xy);
    xy = fliplr(round(xy(t,:)));
    s = mean(ucm(sub2ind2(size(ucm),xy)));
end