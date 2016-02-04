function pt = flip_pt(pt,sz) % FLIP points around x axis for image of given size
pt(:,1) = sz(2)-pt(:,1);
end