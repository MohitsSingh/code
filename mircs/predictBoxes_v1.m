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
Z0 = zeros(size2(I));
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
D = l2(X',XX');
[dist_all,ind_all] = sort(D,2,'ascend');ind_all = ind_all(:,1:min(size(ind_all,2),nn),:)';

all_votes = {};
for t = 1:size(X,2)
    all_votes{t} = getVotes(curBoxCenters(t,:),scales(t),offsets,ind_all(:,t),size2(I));
end
all_votes = cat(1,all_votes{:});
Z_start = accumarray(fliplr(all_votes),ones(size(all_votes,1),1),size2(I));
hsize = 25;
gSize = 5;
F = fspecial('gauss',hsize,gSize);
Z_start = convnFast(Z_start,F,'same');

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
        curBox = boxes(m,:);
        cur_loc = boxCenters(curBox);
        %         cur_x = X(:,m);
        ind = ind_all(:,m);
        curScale = scales(m);
        %         dist = dist_all(:,m);
        cur_votes = getVotes(cur_loc,curScale,offsets,ind,size2(I));
        %         value = values(ind(:));
        %         curValue = [curValue mean(value)];
        weights = ones(size(cur_votes,1),1);
        Z = accumarray(fliplr(cur_votes),weights,size2(I));
        if (useSaliency)
            %Z = Z.*(sal.^.5);
            Z = Z.*sal;
        end
        %Z = imfilter(Z,F);
        Z = convnFast(Z,F,'same');
        %         Z = integgausfilt(Z,5);
        Z0 = Z0 + Z;
        
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
                
                hold on; plotBoxes(curBox);
                plotBoxes(boxes(m,:),'r--','LineWidth',2);
                %         plotPolygons(cur_votes,'r+');
                % visualize top matching patches
                patchesInImage = {};
                for iPatch = 1:min(length(ind),10)
                    imgInd = imgInds(ind(iPatch));
                    curIm = imgs{imgInd};
                    boxInd = subInds(ind(iPatch));
                    patchesInImage{iPatch} = cropper(curIm,all_boxes{imgInd}(boxInd,:));
                end
                vImage = mImage(patchesInImage);
                vl_tightsubplot(plot_m,plot_n,4); imagesc2(vImage);
                xy = repmat(cur_loc,size(cur_votes,1),1);
                vl_tightsubplot(plot_m,plot_n,1);
                vl_tightsubplot(plot_m,plot_n,6);
                V = sc(cat(3,Z_start,I),'prob');
                imagesc2(V);
                vl_tightsubplot(plot_m,plot_n,5);
                V = sc(cat(3,pMap+Z0,I),'prob'); imagesc2(V);
                drawnow; pause(debug_params.pause);
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
function votes = getVotes(cur_loc,curScale,offsets,ind,sz)
curOffsets = offsets(ind,:)/curScale;
% curOffsets = curOffsets.*[all_scales(ind) all_scales(ind)];
cur_votes = bsxfun(@plus,cur_loc,curOffsets);
goods = inImageBounds(sz,cur_votes);
votes = round(cur_votes(goods,:));

function best_match = find_best_match(H,pts,prob)
if (nargin < 3)
    prob = false;
end
if (prob)
    % nms before predicting boxes...
    [subs,vals] = nonMaxSupr( double(H), 3);
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