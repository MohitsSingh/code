function z = visualizeTerm(scores,centers,sz)

if (isscalar(sz))
    sz = [sz sz];
end

ss = [scores centers];
ss = sortrows(ss,1);
ss = unique(ss,'rows','stable');
scores = ss(:,1);
centers = ss(:,2:3);
% [scores,im] = sort(scores,'descend');
% centers = centers(im,:);
[centers,ia] = unique(centers,'rows','stable');
scores = scores(ia);
F = scatteredInterpolant(centers(:,1),centers(:,2),double(scores(:)),'nearest');
[xx,yy] = meshgrid(1:sz(2),1:sz(1));
z = F(xx,yy);
% mesh(xx,yy,z);
% z = interp2(,xx,yy);


