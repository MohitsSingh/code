function [pMap,I,ff_x,ff_y] = predictBoxesRegression(conf,I,features,offsets,kdtree,params,ff_x,ff_y)
nn = params.nn;

s_orig = size2(I);
% I = imResample(I,params.resizeRatio,'bilinear');
I = imrotate(I,params.rot,'bilinear','crop');
if (params.flip)
    I = flip_image(I);
end

origScale = 4;
phow_params = {'Step',params.stepSize,'FloatDescriptors','true','Fast',true,'Sizes',origScale,'Color','gray'};
% phow_params = {'Step',params.stepSize,'FloatDescriptors','true','Fast',true,'Sizes',[2 4 6 8],'Color','gray'};
[F,X,S] = phow_rot(im2single(rgb2gray(I)),0,phow_params{:});

toKeep = ~(sum(X)==0);
F = F(:,toKeep);
X = rootsift(X(:,toKeep));
S = S(:,toKeep);
boxes = inflatebbox([F(1:2,:);F(1:2,:)]',[12 12],'both',true);
curBoxCenters = boxCenters(boxes);

% build a regressor for the x,y offsets.


prm=struct('type','res','loss','L2','eta',.1,'thrr',[0 1],'reg',.1,'S',5,'M',3000,'R',1,'verbose',1);
if (isempty(ff_x))
    [ff_x,ysPrX] = fernsRegTrain(double(features'),offsets(:,1),prm);
    [ff_y,ysPrY] = fernsRegTrain(double(features'),offsets(:,2),prm);
end


[offsets_x,xc] = fernsRegApply(double(X'),ff_x);
[offsets_y,yc] = fernsRegApply(double(X'),ff_y);
centers_x = curBoxCenters(:,1);
centers_y = curBoxCenters(:,2);
votes_x = col(repmat(centers_x',nn,1))+offsets_x;
votes_y = col(repmat(centers_y',nn,1))+offsets_y;
all_votes = round([votes_x votes_y]);
all_goods = inImageBounds(size2(I),all_votes);

% accumulate the mask:
all_weights = ones(size(offsets_x));
Z_start = accumarray(fliplr(all_votes(all_goods,:)),all_weights(all_goods),size2(I));
hSize = 15;
gSize = 2;
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

