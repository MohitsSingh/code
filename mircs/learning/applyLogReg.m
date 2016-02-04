function dets_fixed = applyLogReg(dets,ws,bs)
dets_fixed = dets;
for k = 1:length(ws)
    dets_fixed(k).cluster_locs(:,12) = sigmoid(dets(k).cluster_locs(:,12)*ws(k)+bs(k));
end
end