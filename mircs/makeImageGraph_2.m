function [A] = makeImageGraph_2(G,mask)
[ii,jj] = find(mask);
neighbors = {};
k = 0;

% make sure mask is at least 1 pixel far from image boundary.

mask(:,1) = 0;
mask(:,end) = 0;
mask(1,:) = 0;
mask(end,:) = 0;

[ii,jj] = find(mask);
subs = sub2ind(size(mask),ii,jj);

% I_colors = reshape(vl_xyz2luv(vl_rgb2xyz(G)),[],3);
I_colors = reshape(G,[],3);

v = {};

%rc = [0 1;1 0;1 1;1 -1]; % 8 connected neighborhood
rc = [0 1;1 0]; % 4 connected neighborhood

for irc = 1:size(rc,1)
    r = rc(irc,1);
    c = rc(irc,2);
    n = [ii+r,jj+c];
    cur_subs= sub2ind(size(mask),n(:,1),n(:,2));
    tf = ismember(cur_subs,subs);
    k = k+1;
    neighbors{k} = [subs(tf) cur_subs(tf)];
    v0 =I_colors(neighbors{k}(:,1),:);
    v1 = I_colors(neighbors{k}(:,2),:);
    vv = 1./(1+ sum((v0-v1).^2,2).^.5);
    %abs(v0-v1));
    v{k} = vv;
end

neighbors = cat(1,neighbors{:});
v = im2double(cat(1,v{:}));
m = prod(size(mask));

A = sparse(neighbors(:,1),neighbors(:,2),v,m,m);
A = max(A,A');

end