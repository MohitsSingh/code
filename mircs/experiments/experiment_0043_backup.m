%% experiment 0043 -
%% 16/6/2014
%% desicion process to reach correct location : a patch may point to another location and scale
%% to continue the process. The "winning" value is the correct location in the image: location and scale.


addpath('/home/amirro/code/3rdparty/smallcode/');
%
% X = rand(10000,800);
% [recall, precision] = test(X, 256, 'ITQ');

if (1)
    
    default_init;
    specific_classes_init;
    cls = 1;
    sel_train = class_labels ==cls & isTrain;
    conf.detection.params.detect_min_scale = 1;
    conf.features.winsize = [4 4];
    scaleToPerson = true;
    
    useITQ = true;
    
    % collect many features and the corresp. required center.
    img_h = 100; % we set the image height to a constant size to become somewhat scale invariant,
    % as we current point at a single scale.
    % TODO - also add the normalised location of the patch as a feature.
    % TODO 2: add a larger patch, the "context" of the patch, as a feature.
    % TODO (3,4): add more samples near the target object itself, and near
    % "informative" image regions , and apply several random restarts, and add
    % saliency information...
    
    % TODO - smart restarts, check local maxima as well.
    % Each class seems to have a different effective scale
    % adaptive scales: do cross validation,
    % split data into 50/50 groups and check which scales window size / num.iterations
    % /size of image works the best for each class...
    
    Xs = {};
    offsets = {};
    all_boxes = {}; % locations of boxes in each image
    imgInds = {}; % index of image in current image set.
    subInds = {}; % index of box in relevant image
    imgs = {};
    tinyImages = {};
    all_scales = {};
    
    values = {};
    
    %     imgDataInds = validIndices(sel_train);
    trainInds = validIndices(sel_train);
    for t = 1:length(trainInds)
        %         if (~sel_train(t)),continue;end
        k = trainInds(t)
        %     break
        [I,I_rect] = getImage(conf,newImageData(k));
        if (scaleToPerson)
            scaleFactor = img_h/(I_rect(4)-I_rect(2));
        else
            scaleFactor = img_h/size(I,1);
        end
        I = imResample(I,scaleFactor);
        imgs{t} = im2uint8(I);
        objBox = newImageData(k).obj_bbox*scaleFactor;
        %     imshow(I);
        [X,uus,vvs,scales,~,boxes ] = allFeatures( conf,I,1 );
        
        ovps = boxesOverlap(boxes,objBox);
        %         boxes = [boxes(:,1:4) ovps];
        %         zzz = computeHeatMap(I,boxes,'sum');
        %         figure,imagesc2(zzz);
        
        %         tinyImages{t} = multiCrop(conf,I,boxes(:,1:4));
        bc_obj = boxCenters(objBox);
        %     clf; imagesc2(I); plotBoxes(boxes);
        
        % end
        all_boxes{end+1} = boxes;
        Xs{end+1} = X;
        values{end+1} = ovps;
        subInds{end+1} = col(1:size(boxes,1));
        imgInds{end+1} = t*ones(size(X,2),1);
        offsets{end+1} = bsxfun(@minus,bc_obj,boxCenters(boxes));
        all_scales{end+1} = scales(:);
        %         imgInds{end+1} = k*ones(size(X,2),1);
    end
    XX = cat(2,Xs{:});
    offsets = cat(1,offsets{:});
    all_scales = cat(1,all_scales{:});
    imgInds = cat(1,imgInds{:});
    subInds = cat(1,subInds{:});
    values = cat(1,values{:});
    sel_test = class_labels ==cls & ~isTrain;
    forest = vl_kdtreebuild(XX);
end
%%
conf.detection.params.detect_min_scale =1;

% for t = 2488
for t =1:length(validIndices)
    if (~sel_test(t)),continue;end
    %     if (isTrain(t)),continue;end
    k = validIndices(t)
    %     break
    [I,I_rect] = getImage(conf,newImageData(k));
    if (scaleToPerson)
        scaleFactor = img_h/(I_rect(4)-I_rect(2));
    else
        scaleFactor = img_h/size(I,1);
    end
    I = imResample(I,scaleFactor);
    
    sal = single(foregroundSaliency(conf,newImageData(k).imageID));
    sal = imResample(sal,size2(I));
    %     sal = ones(size2(I));
    
    objBox = newImageData(k).obj_bbox*scaleFactor;
    [X,~,~,~,~,boxes ] = allFeatures( conf,I,1 );
    boxes = boxes(:,1:4);
    
    % select a random location....
    p = randperm(size(X,2));
    curBoxCenters = boxCenters(boxes);
    m = p(1);
    Z0 = zeros(size2(I));
    T = 0;
    plot_m = 2;plot_n = 3;
    alpha_ = .9;
    curValue = 0;
    while(T<50)
        
        curBox = boxes(m,:);
        
        %         nn_big = 1000;
        nn = 500;
        
        cur_loc = boxCenters(curBox);
        cur_x = X(:,m);
        [ind,dist] = vl_kdtreequery(forest,XX,cur_x,'numneighbors',nn,'MaxNumComparisons',0);
        
        
        %         figure,plot(dist);
        
        %D = l2(X(:,m)',XX(:,ind)');
        %         D = l2(X(:,m)',XX');
        %         [dist1,inds1] = sort(D,'ascend');
        %         ind = inds1(1:nn);
        %
        % rerank using dot-product...
        top_x = XX(:,ind);
        prods = X(:,m)'*top_x;
        [u,iu] = sort(prods,'descend');
        %         ind = ind(iu);
        
        curOffsets = offsets(ind,:);
        
        curOffsets = curOffsets.*[all_scales(ind) all_scales(ind)];
        cur_votes = bsxfun(@plus,cur_loc,curOffsets);
        goods = inImageBounds(I,cur_votes);
        cur_votes = round(cur_votes(goods,:));
        value = values(ind(:));
        curValue = [curValue mean(value)];
        weights = ones(size(cur_votes,1),1);
        %         weights = weights.*(1+value(goods));
        Z = accumarray(fliplr(cur_votes),weights,size2(I));
        %         Z = computeHeatMap(I,repmat(curBox,size(curOffsets,1),1)+repmat(curOffsets,1,2));
        Z = Z.*sal;
        Z = imfilter(Z,fspecial('gauss',15,5));
        alpha = (T)/100;
        %                 Z0 = alpha_*Z0+(1-alpha_)*Z;
        % Z0 = Z;
        alpha = (T)/100;
        %         Z0 = alpha*Z0+(1-alpha)*Z;
        Z0 = Z0+Z;
        % Z0 = Z;
        %         Z0 = normalise(Z0);
        
        [max_val,max_val_ind] = max(Z0(:));
        [y,x] = ind2sub(size(Z),max_val_ind);
        center_diff = l2([x y],curBoxCenters);
        [dists,best_match] = min(center_diff);
        next_loc = boxCenters(boxes(best_match,:));
        m = best_match;
        
        % debugging visualization
        if (mod(T,20)==0)
            clf;
            vl_tightsubplot(plot_m,plot_n,3);imagesc2(cropper(I,curBox));
            vl_tightsubplot(plot_m,plot_n,2);imagesc2(Z0);
            
            vl_tightsubplot(plot_m,plot_n,1);
            V = sc(cat(3,Z0,I),'prob');
            vl_tightsubplot(plot_m,plot_n,1); imagesc2(V); hold on; plotBoxes(curBox);
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
            % %             quiver(xy(:,1),xy(:,2),cur_votes(:,1)-xy(:,1),cur_votes(:,2)-xy(:,2));
            % %             plotPolygons(cur_votes,'r+');
            vl_tightsubplot(plot_m,plot_n,6);
            plot(curValue);
            
            drawnow; pause(.01)
            T
        end
        
        
        T = T+1;
    end
    
    %     pause;
    %     bc_obj = boxCenters(objBox);
    %     clf; imagesc2(I); plotBoxes(boxes);
    
    % end
    
    %     Xs{end+1} = X;
    %     offsets{end+1} = bsxfun(@minus,bc_obj,boxCenters(boxes));
end

%     I = imResample(I,.2);
%     r = fhog(im2single(I),8);
%     A = (hogDraw(r.^2,15,1));
%     for k = 1:30:size(X,2)
%         k
%         v{end+1}=hogDraw(reshape(X(:,k),4,4,[]),15,1);
%     end


