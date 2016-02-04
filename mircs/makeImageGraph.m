function [A,pX,pY,dist] = makeImageGraph(G,mask,pStart,pEnd)
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

anyStart = false;
if (nargin > 2)
    
    pStart = round(pStart);
    pEnd = round(pEnd);
    
    
    startMask = zeros(size(mask));
    startMask(pStart(2),pStart(1)) = 1;
    startMask = imdilate(startMask,ones(1,11));
    endMask = zeros(size(mask));
    endMask (pEnd(2),pEnd(1)) = 1;
    endMask  = imdilate(endMask ,ones(1,11));
    
    [indStart] = find(startMask);
    [indEnd] = find(endMask);
  
S = sub2ind(size(mask),pStart(2),pStart(1));
T = sub2ind(size(mask),pEnd(2),pEnd(1));

    
end
v = {};




%rc = [0 1;1 0;1 1;1 -1]; % 8 connected neighborhood
rc = [0 1;1 0]; % 4 connected neighborhood

for irc = 1:size(rc,1)
    %     for r = -1:1
    %         for c = 1:1
    r = rc(irc,1);
    c = rc(irc,2);
    %         if (r==0 && c==0)
    %             continue;
    %         end
    %             if (r==1 && c==1)
    n = [ii+r,jj+c];
    cur_subs= sub2ind(size(mask),n(:,1),n(:,2));
    tf = ismember(cur_subs,subs);
    k = k+1;
    neighbors{k} = [subs(tf) cur_subs(tf)];
    v{k} = abs(G(neighbors{k}(:,1))-G(neighbors{k}(:,2)));
end
% end

neighbors = cat(1,neighbors{:});
v = 1+im2double(cat(1,v{:}));
m = prod(size(mask));

if (anyStart)
    A = sparse(neighbors(:,1),neighbors(:,2),v,m+2,m+2);
else
    A = sparse(neighbors(:,1),neighbors(:,2),v,m,m);
end
%A = max(A,A');

if (anyStart && nargout > 1)
    A(m+1,indStart) = 1;
    A(m+2,indEnd) = 1;
    S = m+1;
    T = m+2;
end
A = max(A,A');

[c r]=ndgrid(1:128,1:128);
%     imshow(G); hold on;
%     gplot(A, [r(:) c(:)]);

if (nargout > 1)
    
    [dist,path,pred] = graphshortestpath(A,S,T);
    if (anyStart)
        [pY,pX] = ind2sub(size(mask),path(2:end-1));
    else
        [pY,pX] = ind2sub(size(mask),path(1:end));
    end
end
%     figure,imshow(G);
%     hold on;
%     [pY,pX] = ind2sub(size(mask),path);
%     plot(pX,pY,'r');

%     [m,im] = min(ii);
%     [mm,imm] = max(jj);
%
%     E = edge(G,'canny');
%     figure,imagesc(G)
%     figure,imagesc(E)

%     figure,imshow(G);hold on;
%         plot(pStart(1),pStart(2),'r+');
%     plot(pEnd(1),pEnd(2),'g+');

%     plot(ii(im),jj(im),'r+');
%     plot(jj(imm),jj(imm),'g+');

%     figure,imshow(mask)


end