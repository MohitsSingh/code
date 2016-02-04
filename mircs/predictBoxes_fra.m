function [pMap,I,roiBox,scaleFactor,Z_masks] = predictBoxes_fra(conf,imgData,XX,params,offsets,all_scales,imgInds,subInds,values,imgs,masks,all_boxes,kdtree,origImgInds,debug_params)

conf.detection.params.detect_min_scale = params.min_scale;
conf.features.winsize = params.wSize;
conf.detection.params.init_params.sbin = params.cellSize;
scaleToPerson = params.scaleToPerson;
useSaliency = params.useSaliency;
img_h = params.img_h; % we set the image height to a constant size to become somewhat scale invariant,
nn = params.nn;
nIter = params.nIter;


if (~isfield(params,'sample_max')) % take max. value at each iteration or do weighted sample.
    params.sample_max = true;
end

if (nargin < 13)
    debug_params.debug = false;
    
end

% roiBox = imgData.faceBox;
% roiBox = round(inflatebbox(roiBox,params.extent,'both',false));
roiParams.infScale = params.extent;
roiParams.absScale = -1;
roiParams.centerOnMouth = params.centerOnMouth;
%roiParams.centerOnMouth = false; %TODO!
[~,roiBox,I] = get_rois_fra(conf,imgData,roiParams);
% [I,I_rect] = getImage(conf,imgData);
% I = cropper(I,roiBox);

I = im2double(I);
scaleFactor = img_h/size(I,1);
origSize = size2(I);
I = imResample(I,scaleFactor,'bilinear');


I = imrotate(I,params.rot,'bilinear','crop');
if (params.flip)
    I = flip_image(I);
end

if (useSaliency)
    sal = single(foregroundSaliency(conf,imgData.imageID));
    sal = cropper(sal,roiBox);
    sal = imResample(sal,size2(I));    
end
% [X,~,~,scales,~,boxes ] = allFeatures( conf,I,1 );

if (strcmp(params.featType,'hog'))
    [X,uus,vvs,scales,~,boxes ] = allFeatures( conf,I,1 );
else
    %[F,X] = vl_dsift(im2single(rgb2gray(I)),'step',1,'FloatDescriptors');
    origScale = 4;
    phow_params = {'Step',params.stepSize,'FloatDescriptors','true','Fast',true,'Sizes',origScale,'Color','gray'};  
%     [F,X] = vl_phow(im2single(rgb2gray(I)),phow_params{:});
%     F = F(1:2,:);
    ff ={};xx  ={};ss = {};
%     for tt = -40:20:40
    for tt = 0
        [F,X,S] = phow_rot(im2single(rgb2gray(I)),tt,phow_params{:});
        toKeep = ~(sum(X)==0);
        F = F(:,toKeep);
        X = X(:,toKeep);
        S = S(:,toKeep);
%         X(1:10,500)'
        ff{end+1} = F;
        xx{end+1} = rootsift(X);
        ss{end+1} = S;
    end
    F = cat(2,ff{:});
    X = cat(2,xx{:});
    S = cat(2,ss{:});
    
    % compute rotated descriptors as well...
    
    boxes = inflatebbox([F;F]',[12 12],'both',true);
    scales = ones(size(X,2),1);
end
        
boxes = boxes(:,1:4);
% select a random location....
curBoxCenters = boxCenters(boxes);
boxCenterInds = sub2ind2(size2(I),fliplr(round(curBoxCenters)));

plot_m = 2;plot_n = 3;
alpha_ = .9;
curValue = 0;

pp = randperm(size(X,2));
pMap = zeros(size2(I));

% find the nearest neighbors once for all patches.
nChecks = 0;
if (params.max_nn_checks > 0)
    nChecks = max(params.max_nn_checks,nn);
end

if (nChecks~=0)
    [ind_all,dist_all] = vl_kdtreequery(kdtree,XX,X,'numneighbors',nn,'MaxNumComparisons',nChecks);
else
    %pdist2
    D = l2(X',XX');
%     D = pdist2(X',XX','sqeuclidean');
    % D = 1-D;
    [dist_all,ind_all] = sort(D,2,'ascend');
    ind_all = ind_all(:,1:min(size(ind_all,2),nn),:)';
    dist_all = dist_all(:,1:min(size(dist_all,2),nn),:)';
end



voteImgInds = origImgInds(ind_all);
if (debug_params.keepAllVotes)
    keepVotes = true(size(voteImgInds));
else
    keepVotes = voteImgInds ~= imgData.imgIndex;
end


% neighbor visualization mode.
% % % clf; subplot(1,3,1); imagesc2(I);
% % % while (true);
% % %     
% % % %     plotPolygons(curBoxCenters,'r.');
% % %     [x,y,b] =ginput(1);
% % %     [d] = l2([x y],curBoxCenters);
% % %     [m,im] = min(d);
% % %     curPatch = cropper(I,round(boxes(im,:)));
% % %     
% % %     curInds = ind_all(:,im);
% % %     for iPatch = 1:min(4,length(curInds))
% % %         imgInd = imgInds(curInds(iPatch));
% % %         curIm = imgs{imgInd};
% % %         boxInd = subInds(curInds(iPatch));
% % %         patchesInImage{iPatch} = cropper(curIm,round(all_boxes{imgInd}(boxInd,:)));
% % %     end
% % %     
% % %     clf; subplot(1,3,1); imagesc2(I);
% % %     subplot(1,3,2);imagesc2(curPatch);
% % %     subplot(1,3,3); MM = multiImage(patchesInImage,false); imagesc2(MM);
% % % end

S = repmat(origScale./S,size(ind_all,1),1);
offsets_x = offsets(ind_all,1).*S(:);
offsets_y = offsets(ind_all,2).*S(:);
centers_x = curBoxCenters(:,1);
centers_y = curBoxCenters(:,2);
votes_x = col(repmat(centers_x',nn,1))+offsets_x;
votes_y = col(repmat(centers_y',nn,1))+offsets_y;
all_votes = round([votes_x votes_y]);
all_goods = inImageBounds(size2(I),all_votes) & keepVotes(:);
weights = exp(-dist_all(:)*10);
%all_goods(~keepVotes) = false;
% weights = weights.*keepVotes(:);

% accumulate the mask:
pad_size = 0;
Z_masks = false(size2(I)+pad_size*2);
votingImageInds = imgInds(ind_all);

% offsets_x = round(offsets_x);
% offsets_y = round(offsets_y);
% imbb = [1 1  fliplr(size2(I))];
% 

% U = zeros(size2(I));
% U(31,31) = 1;
% figure,imagesc2(curMask)
% UU = convnFast(curMask,U,'same');
% figure,imagesc2(UU-curMask)
% figure,imagesc2(UU)

% % % % % 

if (nargout == 5)
    
    [uniqueMasks,ia,ib] = unique(votingImageInds);
    for iUniqueMasks = 1:length(uniqueMasks)
        curMaskInd = uniqueMasks(iUniqueMasks);
        sel_vote = votingImageInds(:)==curMaskInd & all_goods;
        curMask = masks{curMaskInd};
        if (none(curMask))
            continue;
        end
        curMask = double(cropper(curMask,region2Box(curMask)));
        %     [y,x] = find(curMask);
        %     y = mean(y);
        %     rx = mean(x);
        %curVotes = round([size(Z_masks,2)/2+offsets_x(sel_vote), size(Z_masks,1)/2+offsets_y(sel_vote)]);
        curVotes = all_votes(sel_vote,:);
        if (none(curVotes))
            continue
        end
        %     curGoods = inImageBounds(Z_masks,curVotes);
        voteMap = zeros(size(Z_masks));
        voteMap = accumarray(fliplr(curVotes),ones(nnz(sel_vote),1),size2(Z_masks));
        voteMap = convnFast(voteMap,curMask,'same');
        Z_masks = Z_masks+voteMap;
        
        %voteMap = griddata(offsets_x(sel_vote),offsets_y(sel_vote),ones(nnz(sel_vote),1),X,Y);
    end
end
%Z_masks = Z_masks(pad_size/2:end,pad_size/2:end);
%Z_masks = Z_masks(1:size(I,1),1:size(I,2));

% for iVote = 1:length(votingImageInds)
%     
%     
%     
%     curMask = masks{votingImageInds(iVote)};
%     curMask = shiftRegions(curMask,
%         shiftRegions(masks(voting
%     startPt = [offsets_x(iVote) offsets_y(iVote)];
%     endPt = 
% end


%all_weights = ones(size(all_goods));
all_weights = ones(size(weights));
% all_weights = weights;

Z_start = accumarray(fliplr(all_votes(all_goods,:)),all_weights(all_goods),size2(I));

% the vote map given by each patch...

%hsize = round(size(I,1)/7);
hSize = 15;
% gSize = 5;
gSize = 4;

F = fspecial('gauss',hSize,gSize);
Z_start = convnFast(Z_start,F,'same');
if (nIter==0)
    pMap = Z_start;
    if (useSaliency)
            pMap = pMap.*sal;
        end
%     pMap = imResample(pMap,size2(I));
    return;
end
Z0 = Z_start;
if (debug_params.debug && debug_params.doVideo)
    outputVideo= VideoWriter(fullfile('/home/amirro/notes/images/2014_06_19/',[imgData.imageID '.avi'])...
        );
    outputVideo.FrameRate = 4;
    outputVideo.open;
end
for iRestart = 1
    %     m = pp(iRestart);
    
    m = find_best_match(Z_start,curBoxCenters);
    
    T = 0;
    while(T<nIter)
        
        curBox = round(boxes(m,:));
        cur_loc = boxCenters(curBox);
        ind = ind_all(:,m);
        curX = X(:,m);
        dist_ = dist_all(:,m);
        curScale = scales(m);
        [cur_votes,goodInds] = getVotes(cur_loc,curScale,offsets,ind,size2(I));        
        if (~params.voteAll)
            weights = normalise(exp(-dist_(goodInds)));
            
            Z = accumarray(fliplr(cur_votes(goodInds,:)),weights,size2(I));
        else
            voteWeights = Z0(boxCenterInds);
            for t = 1:length(voteWeights)
                all_weights{t} = voteWeights(t)*ones(size(all_weights{t},1),1);
            end
            cur_weights = cat(1,all_weights{:});

%             Z =  accumarray(fliplr(all_votes(all_goods,:)),ones(size(all_votes(all_goods,:),1),1),size2(I));
            Z =  accumarray(fliplr(all_votes(all_goods,:)),cur_weights(all_goods),size2(I));
            % sample the vote map at the box centers
        end
        if (useSaliency)
            Z = Z.*sal;
        end
        Z = convnFast(Z,F,'same');
        %         Z = integgausfilt(Z,5);
        Z0 = Z0 + Z;
        % Z0 = Z;
        
        %         Z0 = normalise(Z0);%+normalise(Z_start))/2;
        
        best_match = find_best_match(Z0,curBoxCenters,~params.sample_max);
        m = best_match;
        
        % debugging visualization
        if (debug_params.debug)
            if (mod(T,debug_params.showFreq)==0)
                clf;
                vl_tightsubplot(plot_m,plot_n,3);imagesc2(cropper(I,curBox));
                vl_tightsubplot(plot_m,plot_n,2);imagesc2(Z0);
                
                vl_tightsubplot(plot_m,plot_n,1);
                V = sc(cat(3,Z0,I),'prob');
                vl_tightsubplot(plot_m,plot_n,1); imagesc2(V);
                if (debug_params.doVideo)
                    writeVideo(outputVideo,V);
                end
                
                hold on; plotBoxes(curBox,'g-','LineWidth',2);
                plotBoxes(boxes(m,:),'r--','LineWidth',2);
                %         plotPolygons(cur_votes,'r+');
                % visualize top matching patches
                patchesInImage = {};
                toShow = min(10,length(ind));
                for iPatch = 1:toShow
                    imgInd = imgInds(ind(iPatch));
                    curIm = imgs{imgInd};
                    boxInd = subInds(ind(iPatch));
                    
                    %                     [X_,uus_,vvs_,scales_,~,boxes_] = allFeatures( conf,curIm,1 );
                    %                     D2 = l2(curX',X_');
                    %                     [q,iq]= min(D2);
                    
                    patchesInImage{iPatch} = cropper(curIm,round(all_boxes{imgInd}(boxInd,:)));
                end
                vImage = mImage(patchesInImage);
                vl_tightsubplot(plot_m,plot_n,4); imagesc2(vImage);
%                 vl_tightsubplot(plot_m,plot_n,5); imagesc2(sal);
                
                drawnow;
                if (0)
                    xy = repmat(cur_loc,size(cur_votes,1),1);
                    vl_tightsubplot(plot_m,plot_n,1);
                    dd = cur_votes-xy;
                    quiver(xy(1:toShow,1),xy(1:toShow,2),dd(1:toShow,1),dd(1:toShow,2),0,'r');
                    vl_tightsubplot(plot_m,plot_n,6);
                    V = sc(cat(3,Z_start,I),'prob');
                    imagesc2(V);
                    vl_tightsubplot(plot_m,plot_n,5);
                    V = sc(cat(3,pMap+Z0,I),'prob'); imagesc2(V);
                end
                if (debug_params.pause==0)
                    pause;
                else
                    pause(debug_params.pause);
                end
                T
            end
        end
        
        T = T+1;
    end
    pMap = Z0 +pMap;
end
% pMap = imResample(Z0,origSize);
if(debug_params.debug && debug_params.doVideo)
    close(outputVideo);
end
function [votes,goods] = getVotes(cur_loc,curScale,offsets,ind,sz)
% curScale = 1;
curOffsets = offsets(ind,:)/curScale;
% curOffsets = curOffsets.*[all_scales(ind) all_scales(ind)];
votes = round(bsxfun(@plus,cur_loc,curOffsets));
goods = inImageBounds(sz,votes);

function best_match = find_best_match(H,pts,prob)
if (nargin < 3)
    prob = false;
end
if (prob)
    % nms before predicting boxes...
    [subs,vals] = nonMaxSupr( double(H), 10,[],3);
    ff = vals>0;
    vals = vals(ff);subs = subs(ff,:);
    v = weightedSample(1:length(vals),vals,1);
    max_val_ind = sub2ind2(size2(H),subs(v,:));
else
    [max_val,max_val_ind] = max(H(:));
end

[y,x] = ind2sub(size(H),max_val_ind);
center_diff = l2(double([x y]),pts);
[dists,best_match] = min(center_diff);

function [F,D,S] = phow_rot(I,rot,varargin)
% I1 = I; 
    I = imrotate(I,rot,'bilinear','crop');
    [F,D] = vl_phow(I,varargin{:});            
    S = F(4,:);
    F = F(1:2,:);
    F = rotate_pts(F(1:2,:)',-pi*rot/180,size2(I)/2)';
    
