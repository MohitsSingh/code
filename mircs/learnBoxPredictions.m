function [poseletPreds,probMaps] = learnBoxPredictions(conf,action_rois)
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
conf.max_image_size = inf;
conf.get_full_image = true;

nPoselets = length(conf.poseletModel.fg_masks);

% learn the gaussian for each poselet...
f_train = find(train_labels);


poseletPreds = cell(length(conf.poseletModel.fg_masks),1);
debug_ = false;
for iTrain = 1:length(f_train)
    iTrain
    curID = train_ids{f_train(iTrain)};
    
    [~,xmin,ymin,xmax,ymax ] = toImage(conf, curID,true,true);
    
    [rects,rects_poselets,poselet_centers,poselet_ids,s,is] = getPoseletData(conf,curID,...
        xmin,ymin,xmax,ymax);
    
    sizes = rects_poselets(:,3);
    
    action_box = action_rois(iTrain,:);
    
    normalized_boxes = bsxfun(@minus,action_box,[poselet_centers poselet_centers]);
    normalized_boxes = bsxfun(@rdivide,normalized_boxes,sizes);
    
    for pID = 1:length(poselet_ids)
        poseletPreds{poselet_ids(pID)} = [poseletPreds{poselet_ids(pID)};normalized_boxes(pID,:)];
    end
    %     conf.poseletModel.fg_masks{poselet_ids(1)}
    
    
    if (debug_)
        img = getImage(conf,curID);
        close all;
        figure,imshow(img);hold on;
        
        plotBoxes2(rects(is(1),[2 1 4 3]),'g','LineWidth',2);
        
        plotBoxes2(rects_poselets(:,[2 1 4 3]),'m','LineWidth',1);
        plotBoxes2([ymin xmin ymax xmax],'--b','LineWidth',3);
        
        plotBoxes2( action_rois(iTrain,[2 1 4 3]),'y','LineWidth',3)
        legend('predicted box','poselets','person box','action box');
    end
end

f_test = find(test_labels);
probMaps = cell(1,length(f_test));
debug_ = true;
for iTest = 1:length(f_test)
    iTest
    curID = test_ids{f_test(iTest)};
    
    [~,xmin,ymin,xmax,ymax ] = toImage(conf, curID,true,true);
    [rects,rects_poselets,poselet_centers,poselet_ids,s,is] = getPoseletData(conf,curID,...
        xmin,ymin,xmax,ymax);
    
    sizes = rects_poselets(:,3);
    
    rects_r = cell(size(poseletPreds));
    
    img = getImage(conf,curID);
    
    bc = boxCenters(rects_poselets);
    bc = [bc bc];
    Z = zeros(size(img,1),size(img,2));
    allBoxes = {};
    for t = 1:length(poselet_ids)
        allBoxes{t} = [];
        if (~isempty(poseletPreds{poselet_ids(t)}))
            %             t
            p = poseletPreds{poselet_ids(t)}*sizes(t);
            allBoxes{t} = bsxfun(@plus,bc(t,:),p);
            %             figure(1);hold on; plotBoxes2(bsxfun(@plus,bc(t,:),p))
            p(:,[1 3]) = p(:,[1 3])*-1;
            p = bsxfun(@plus,bc(t,:),p);
            p = p(:,[3 2 1 4]);
            %             figure(1);hold on; plotBoxes2(bsxfun(@plus,bc(t,:),p),'r');
            allBoxes{t} = [allBoxes{t};p];
        end
        %         plotBoxes2(a(:,[2 1 4 3]));
    end
    
    close all;
    allBoxes = cat(1,allBoxes{:});
    clf,subplot(1,2,1);
    if size(allBoxes,1)==0
        warning(['no boxes found for image ' curID]);
        continue;
    end
    Z = drawBoxes(imresize(img,1),allBoxes,[],2);
    if (debug_)
        
        %imagesc(img);axis image;hold on;plotBoxes2(allBoxes(:,[2 1 4 3]),'-+');
        imagesc2(img);
        %     imagesc(img);axis image;
        
        subplot(1,2,2);
        Z = drawBoxes(imresize(img,1),allBoxes,[],2);
        Z = Z{1};
        ZZ=sc(cat(3, Z, img), 'prob');
        imagesc(ZZ);axis image;
        
        
        
        pause;
        
    end
    
    probMaps{iTest} = Z;
    %
    if (debug_)
        img = getImage(conf,curID);
        close all;
        figure,imshow(img);hold on;
        
        plotBoxes2(rects(is(1),[2 1 4 3]),'g','LineWidth',2);
        
        plotBoxes2(rects_poselets(:,[2 1 4 3]),'m','LineWidth',1);
        plotBoxes2([ymin xmin ymax xmax],'--b','LineWidth',3);
        
        plotBoxes2( action_rois(iTrain,[2 1 4 3]),'y','LineWidth',3)
        legend('predicted box','poselets','person box','action box');
    end
end

% now check again on the train!!


end



% b = 1;
%figure,hold on; plotBoxes2(poseletPreds{3}(:,[2 1 4 3]))