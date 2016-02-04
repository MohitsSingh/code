function rprops = getParallelLines(rprops,Z)
% figure; hold on;
for k = 1:length(rprops)
    curBoundary = rprops(k).perim;
    tol = 2;
    seglist = lineseg({curBoundary}, tol);
    
    segs = seglist2segs(seglist);
    vecs = segs2vecs(segs);
    [X,norms] = normalize_vec(vecs');
    cos_angles = X'*X;
    % remove self-angle
    cos_angles = cos_angles.*(1-eye(size(cos_angles)));
    maxAngle = 20; % maximal angle between adjacent segments.
    [ii,jj] = find(abs(cos_angles) >= cosd(maxAngle)); % ii,jj are possible pairs of segments.
    rprops(k).parallelSegs = segs(ii,:);
    % find if there is a couple of parallel edges
            drawedgelist(seglist, size(Z), 2, 'rand');
end
end