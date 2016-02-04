function labels = fixLabels(labels)
    uLabels = unique(labels(:));
    maxLabel = max(uLabels);
    for k = 1:length(uLabels)
        bw = labels == uLabels(k);
        r = bwlabel(bw,4);
        u = unique(r(r~=0));
        for kk = 2:length(u)
            maxLabel = maxLabel+1;
            labels(r ==u(kk)) = maxLabel;
        end
    end
end