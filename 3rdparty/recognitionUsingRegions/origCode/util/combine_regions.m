function regions = combine_regions(imgFile, th)
% function regions = combine_regions(imgFile, th)
%
% This function inputs image filename and returns the regions extracted
% from the image UCM signal.
%
% Copyright @ Chunhui Gu, April, 2009

if (nargin < 2)
    th = 40;
end

[ucm,label] = img2ucmlabel(imgFile,th);
[regionlabels,regionareas,connectedmatrix] = get_topology(ucm,label);
numregions = length(regionlabels);

if numregions == 1,
    reg.regionlabels = regionlabels;
end;

while length(regionlabels) > 1,

    if ~exist('rId','var') || rId == 1, % finest layer
        % unary region
        for rId = 1:numregions,
            reg(rId).regionlabels = regionlabels(rId);
            reg(rId).ucmlevel = th;
        end;
        
    else % update layers

        %imshow(connectedmatrix .* (connectedmatrix < Inf));
        %title(['Dim is ' num2str(size(connectedmatrix,1))]);

        [ri,rj,connectedmatrix,ucmlevel] = update_connectedmatrix(connectedmatrix);

        if ri == rj,
            break;
        end;

        % unary region
        rId = rId + 1;
        reg(rId).regionlabels = [reg(regionlabels(ri)).regionlabels, reg(regionlabels(rj)).regionlabels];
        reg(rId).ucmlevel = ucmlevel;

        regionlabels = [regionlabels, rId];
        regionlabels([ri,rj]) = [];

    end;
end;

reg_all = reg;

rlabel = cell(1,numregions);
for ii = 1:numregions,
    rlabel{ii} = find(label==ii);
end;

regions = cell(1,length(reg_all));
for ii = 1:length(reg_all),
    regions{ii} = false(size(label));
    for rr = 1:length(reg_all(ii).regionlabels),
        regions{ii}(rlabel{reg_all(ii).regionlabels(rr)}) = true;
    end;
end;

%%%%%%%%%%%%%%%%%%%%%
function [ucm,label] = img2ucmlabel(imgFile,th)

ucm2 = img2ucm2(imgFile);
ucm = ucm22ucm(ucm2, -1, [imgFile '.tmp']);
ucm = ucm .* (ucm >= th);

uniq = unique(ucm);
if length(uniq) == 1,
    label = ones(size(ucm));
else
    label = ucm22label(ucm2, uniq(2));
end;

%%%%%%%%%%%%%%%%%%%%%
function [regionlabels,regionareas,connectedmatrix] = get_topology(ucm,label)

regionlabels = unique(label)';
numregions = length(regionlabels);

regionareas = zeros(1,numregions);
for ii = 1:numregions,
    regionareas(ii) = sum(sum(label==regionlabels(ii)));
end;

connectedmatrix = ones(numregions,numregions)*Inf;
[h,w] = size(ucm);
[py,px] = find(ucm > 0);
for ii = 1:length(px),
    neighbory = max(1,py(ii)-1):min(h,py(ii)+1);
    neighborx = max(1,px(ii)-1):min(w,px(ii)+1);
    s = unique(label(neighbory,neighborx));
    if length(s) == 2,
        connectedmatrix(s(1),s(2)) = ucm(py(ii),px(ii));
        connectedmatrix(s(2),s(1)) = ucm(py(ii),px(ii));
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%
function [ri,rj,connectedmatrix,ucmlevel] = update_connectedmatrix(connectedmatrix)

ucmlevel = min(connectedmatrix(:));
[ri,rj] = find(connectedmatrix == min(connectedmatrix(:)));
ri = ri(1); rj = rj(1);
numregions = size(connectedmatrix,1);
for ii = 1:numregions,
    connectedmatrix(numregions+1,ii) = min(connectedmatrix([ri;rj],ii));
    connectedmatrix(ii,numregions+1) = min(connectedmatrix(ii,[ri,rj]));
end;
connectedmatrix(end,end) = Inf;
connectedmatrix([ri;rj],:) = [];
connectedmatrix(:,[ri,rj]) = [];