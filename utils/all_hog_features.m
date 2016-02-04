function [X,uus,vvs,scales,t,boxes ] = all_hog_features( winSize,sbin,I,ovp ,esvm_params)
%ALLFEATURES Summary of this function goes here
%   Detailed explanation goes here
t = get_pyramid(I,esvm_params);
S = winSize;
% fsize = conf.detection.params.init_params.features();
fsize = 31;
% S = [S S];

if (nargin < 3) % maximal allowed overlap between patches.
    ovp = 1;
end

% when requiring an overlap that creates a fractional delta between
% adjacent frames, floor it to be on the safe side.
delta = max(1,floor(S*(1-ovp)/(1+ovp)));

pyr_N = cellfun(@(x)prod(max(0,[size(x,1) size(x,2)]-S+1)),t.hog);
if (sum(pyr_N) == 0)
    %     warning('pyr_N is 0, returing empty everything');
    X = [];
    uus = [];
    vvs = [];
    scales = [];
    t = [];
    boxes = [];
    return;
end

t.hog = t.hog(pyr_N > 0);
scales = t.scales(pyr_N > 0);
sumN = sum(pyr_N);

X = zeros(S(1)*S(2)*fsize,sumN);
offsets = cell(length(t.hog), 1);
uus = cell(length(t.hog),1);
vvs = cell(length(t.hog),1);

prevCount = 0;
currentCount = 0;
bsizes = [];
for i = 1:length(t.hog)
    s = size(t.hog{i});
    NW = s(1)*s(2);
    ppp = reshape(1:NW,s(1),s(2));
    %     ppp_sub = ppp(1:delta(1):end,1:delta(2):end);
    %     z = zeros(s(1),s(2));
    %     z(ppp_sub) = 1;
    %     z_ = zeros(s(1),s(2));
    %     z_(b(1,:)) = 1;
    %     figure,imshow(z_);
    %     figure,imshow(z);
    
    curf = reshape(t.hog{i},[],fsize);
    b = im2col(ppp,[S(1) S(2)]);
    %     ppp_sub=ppp_sub(ppp_sub<=length(b));
    %     b = b(:,ppp_sub);
    offsets{i} = b(1,:);
    offsets{i}(end+1,:) = i;
    bsizes = cat(2,bsizes,size(b,2));
    T = zeros(S(1)*S(2)*fsize,size(b,2),'single');
    
    for j = 1:size(b,2)
        %         j
        T(:,j) = reshape (curf(b(:,j),:),[],1);
    end
    
    ppp_sub = col(ppp(1:delta(1):end,1:delta(2):end));
    %ppp_sub = ppp_sub(ppp_sub<=max(offsets{i}(1,:)));
    [c, ia, ib] = intersect(ppp_sub,offsets{i}(1,:));
    offsets{i} = offsets{i}(:,ib);
    [uus{i},vvs{i}] = ind2sub(s,offsets{i}(1,:));
    currentCount = prevCount + length(ib);
    X(:,prevCount+1:currentCount) = T(:,ib);
    prevCount = currentCount;
end

X = X(:,1:currentCount);

% X = X(:,1:counter);
offsets = cat(2,offsets{:});
scales = scales(offsets(2,:));
uus = cat(2,uus{:});
vvs = cat(2,vvs{:});

if (nargout == 6)
    boxes = uv2boxes(sbin,uus,vvs,scales,t,S);
end
end

