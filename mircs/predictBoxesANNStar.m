function [pMap,I] = predictBoxesANNStar(conf,I,features,offsets,kdtree,params,featureBox)
nn = params.nn;
if (nargin < 7)
    featureBox = [1 1 size(I,2) size(I,1)];
end
s_orig = size2(I);
% I = imResample(I,params.resizeRatio,'bilinear');
I = imrotate(I,params.rot,'bilinear','crop');
if (params.flip)
    I = flip_image(I);
end

origScale = 4;
phow_params = {'step',params.stepSize,'floatdescriptors','true','fast',true,'sizes',origScale,'color','gray'};
% phow_params = {'Step',params.stepSize,'FloatDescriptors','true','Fast',true,'Sizes',[2 4 6 8],'Color','gray'};
[F,X,S] = phow_rot(im2single(rgb2gray(I)),0,phow_params{:});

toKeep = ~(sum(X)==0);
toKeep = toKeep & inBox(featureBox(1:4),F')';
F = F(:,toKeep);
X = rootsift(X(:,toKeep));
S = S(:,toKeep);
boxes = inflatebbox([F(1:2,:);F(1:2,:)]',[12 12],'both',true);
curBoxCenters = boxCenters(boxes);

% find the nearest neighbors once for all patches.
nChecks = 0;
if (params.max_nn_checks > 0)
    nChecks = max(params.max_nn_checks,nn);
end
if (nChecks~=0 && ~isempty(kdtree))
    [ind_all,dist_all] = vl_kdtreequery(kdtree,features,X,'numneighbors',nn,'MaxNumComparisons',nChecks);
else
    D = l2(X',features');
    [dist_all,ind_all] = sort(D,2,'ascend');
    ind_all = ind_all(:,1:min(size(ind_all,2),nn),:)';
    dist_all = dist_all(:,1:min(size(dist_all,2),nn),:)';
end

S = repmat(origScale./S,size(ind_all,1),1);
%S = repmat(S/origScale,size(ind_all,1),1);
% S = ones(size(S)); %TODO!!
offsets_x = offsets(ind_all,1).*S(:);
offsets_y = offsets(ind_all,2).*S(:);
centers_x = curBoxCenters(:,1);
centers_y = curBoxCenters(:,2);
votes_x = col(repmat(centers_x',nn,1))+offsets_x;
votes_y = col(repmat(centers_y',nn,1))+offsets_y;
all_votes = round([votes_x votes_y]);
all_goods = inImageBounds(size2(I),all_votes);
weights = exp(-dist_all(:)*100);
weights = ones(size(weights));

% x2(I); 
% hold on;
% quiver(F(1,:)',F(2,:)',offsets_x,offsets_y);

% accumulate the mask:
all_weights = weights;
%all_weights = ones(size(weights));
if (any(all_goods))
    Z_start = accumarray(fliplr(all_votes(all_goods,:)),all_weights(all_goods),size2(I));
else
    Z_start = zeros(size2(I));
end
hSize = 5;
% gSize = 2;
gSize = .5;
F = fspecial('gauss',hSize,gSize);
Z_start = convnFast(Z_start,F,'same');

pMap = Z_start;

if (params.flip)
    pMap = flip_image(pMap);
end

pMap = imrotate(pMap,-params.rot,'bilinear','crop');
% pMap = imResample(pMap,s_orig,'bilinear');

function [F,D,S] = phow_rot(I,rot,varargin)
% I1 = I;
I = imrotate(I,rot,'bilinear','crop');
[F,D] = vl_phow(I,varargin{:});
S = F(4,:);
F = F(1:2,:);
F = rotate_pts(F(1:2,:)',-pi*rot/180,size2(I)/2)';

