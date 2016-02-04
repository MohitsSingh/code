function pMap = predictBoxes(conf,imgData,XX,params,offsets,all_scales,imgInds,subInds,values,imgs,all_boxes,kdtree,debug_params)

conf.detection.params.detect_min_scale = params.min_scale;
conf.features.winsize = params.wSize;
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

[I,I_rect] = getImage(conf,imgData);
origSize = size2(I);
if (scaleToPerson)
    scaleFactor = img_h/(I_rect(4)-I_rect(2));
    
    if (params.normalizeWithFace)
        the_face = imgData.alternative_face;
        if (isempty(the_face))
            the_face=imgData.faceBox;
        end
        scaleFactor = .2*img_h/(the_face(4)-the_face(2));
    end
    
else
    scaleFactor = img_h/size(I,1);
end
I = imResample(I,scaleFactor);
if (useSaliency)
    sal = single(foregroundSaliency(conf,imgData.imageID));
    sal = imResample(sal,size2(I));
end
[X,~,~,scales,~,boxes ] = allFeatures( conf,I,1 );
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
    nChecks = max(params.max_nn_checks,nn)
end

% [ind_all,dist_all] = vl_kdtreequery(kdtree,XX,X,'numneighbors',nn,'MaxNumComparisons',nChecks);
%pdist2
% D2 = l2(X',XX');
D = pdist2(X',XX','cosine');
% D = 1-D;
[dist_all,ind_all] = sort(D,2,'ascend');
ind_all = ind_all(:,1:min(size(ind_all,2),nn),:)';
dist_all = dist_all(:,1:min(size(dist_all,2),nn),:)';


all_votes = {};all_goods = {};
all_weights = {};
for t = 1:size(X,2)
    [all_votes{t},all_goods{t}] = getVotes(curBoxCenters(t,:),scales(t),offsets,ind_all(:,t),size2(I));
    all_weights{t} = ones(size(all_votes{t},1),1);
    
end
all_goods = cat(1,all_goods{:});
all_votes = cat(1,all_votes{:});
Z_start = accumarray(fliplr(all_votes(all_goods,:)),ones(size(all_votes(all_goods,:),1),1),size2(I));
% Z0 = zeros(size2(I));

% the vote map given by each patch...

%hsize = round(size(I,1)/7);
hSize = 15;
gSize = 5;
F = fspecial('gauss',hSize,gSize);
Z_start = convnFast(Z_start,F,'same');
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
%         ind = ind(goodInds);
%         dist_ = dist_(goodInds);
        weights = normalise(exp(-dist_(goodInds)));
        weights = ones(size(weights));
        
        
        if (~params.voteAll)
            Z = accumarray(fliplr(cur_votes(goodInds,:)),weights,size2(I));
        else
            voteWeights = Z0(boxCenterInds);
            for t = 1:length(voteWeights)
                all_weights{t} = voteWeights(t)*ones(size(all_weights{t},1),1);
            end
%             Z = accumarray(fliplr(all_votes),cat(1,all_weights{:}),size2(I));
               Z =  accumarray(fliplr(all_votes(all_goods,:)),ones(size(all_votes(all_goods,:),1),1),size2(I));
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
                    
                    patchesInImage{iPatch} = cropper(curIm,all_boxes{imgInd}(boxInd,:));
                end
                vImage = mImage(patchesInImage);
                vl_tightsubplot(plot_m,plot_n,4); imagesc2(vImage);
                
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
                pause(debug_params.pause);
                T
            end
        end
        
        T = T+1;
    end
    pMap = Z0 +pMap;
end
pMap = imResample(Z0,origSize);
if(debug_params.debug && debug_params.doVideo)
    close(outputVideo);
end
function [votes,goods] = getVotes(cur_loc,curScale,offsets,ind,sz)
curScale = 1;
curOffsets = offsets(ind,:)/curScale;
% curOffsets = curOffsets.*[all_scales(ind) all_scales(ind)];
cur_votes = bsxfun(@plus,cur_loc,curOffsets);
goods = inImageBounds(sz,cur_votes);
votes = round(cur_votes);

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