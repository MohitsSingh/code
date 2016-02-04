function [Zs,pts] = createConsistencyMaps(dets,sz,imgSel,maxSamples,toBlur)
if (~isstruct(dets)) 
    dets_ = struct;
    dets(:,11) = 0;
    dets(:,12) = 0;
    dets_.cluster_locs = dets;
    dets = dets_;
end
Zs = cell(1,length(dets));
pts = cell(1,length(dets));
if (nargin < 4)
    maxSamples = inf;
end
for k = 1:length(dets)
    Z = zeros(sz(1:2));
    curLocs = dets(k).cluster_locs;
    if (~isempty(imgSel))
        [a,b,c] = intersect(imgSel,curLocs(:,11));
    else
        c = 1:length(curLocs(:,11));
    end
    c= c(1:min(length(c),maxSamples));
    %Zs{k}=drawBoxes(Z,curLocs(c,:),[],2);
    ff = find(curLocs(c,7));
    curLocs(ff,:) = flip_box(curLocs(ff,:),sz);
    
    r = [boxCenters(curLocs(c,:)),curLocs(c,8), curLocs(c,12)];
    pts{k} = r;
    bc = round(pts{k});
    inds_ = (sub2ind(sz,bc(:,2),bc(:,1)));
    zz = zeros(sz);
    m = min(curLocs(c,12));
    
    for q = 1:length(inds_)
        zz(inds_(q)) = zz(inds_(q))+1;%(curLocs(c(q),12)-m);
    end
    Zs{k} = zz;
    
    if (toBlur)
        Zs{k} = imfilter(Zs{k},fspecial('gauss',toBlur(1),toBlur(2)));
        Zs{k} = (Zs{k}/max(Zs{k}(:)));
    end
    %     Zs{k}=drawBoxes(Z,curLocs(c,:),[],2);
    %     Zs{k} = Zs{k}/length(c);
end
end
