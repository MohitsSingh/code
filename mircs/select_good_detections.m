function newRecs = select_good_detections(all_rects,recs,cls,n,imgDir)

ind = all_rects(:,6);
true_bbs = {};
all_clsinds = {};
for k = 1:length(recs)
    clsinds=strmatch(cls,{recs(k).objects(:).class},'exact');
    all_clsinds{k} = clsinds;
    true_bb = cat(1,recs(k).objects(clsinds).bbox);
    %true_bbs{k} = [true_bb,ones(size(true_bb,1),1)*k];
    true_bbs{k} = true_bb;
    %curDets = all_rects(ind==k,:);    
end

[s,is] = sort(all_rects(:,5),'descend');

% for each detection, retain the true bounding box with the highest
% overlap.

visited = false(size(recs));
newRecs = recs;
for k = 1:length(newRecs)
    newRecs(k).objects = [];
end
for t = 1:min(n,length(s))
    k = is(t);
    curRect = all_rects(k,1:4);
    rec_ind = all_rects(k,6);
    if (visited(rec_ind))
%         continue;
    end    
    ovp = boxesOverlap(true_bbs{rec_ind},curRect);
    [m,im] = max(ovp);
    if (m > .5)
        toAdd = recs(rec_ind).objects(all_clsinds{rec_ind}(im));
        wasAdded = false;
        if (~isempty(newRecs(rec_ind).objects))
            added = cat(1,newRecs(rec_ind).objects.bbox);
            toAddBoxes = cat(1,toAdd.bbox);        
            if (max(boxesOverlap(added,toAddBoxes))<1)
                newRecs(rec_ind).objects = [newRecs(rec_ind).objects,toAdd];
                wasAdded = true;
            end
        else
            newRecs(rec_ind).objects = [newRecs(rec_ind).objects,toAdd];
            wasAdded = true;
        end
        % make sure there is no bounding box duplication.
        if (wasAdded)
        % show the detection and chosen record.
            I = imread(fullfile(imgDir,[recs(rec_ind).filename '.JPEG']));
            clf; imagesc(I); axis image; hold on;
            plotBoxes(curRect,'m--','LineWidth',2);
            plotBoxes(cat(1,newRecs(rec_ind).objects.bbox),'g','LineWidth',2);
            drawnow
        end
%         pause;
        visited(rec_ind) = true;
    end    
end
newRecs(~visited) = [];
% 
%     I = imread(fullfile(imgDir,catDir,d(all_rects(is(k),6)).name));
%     I = im2double(I);
%     rects = detect(I,curClassifier.weights,curClassifier.bias,curClassifier.object_sz,cell_size,features,detection,threshold);
%     if (isempty(rects))
%         continue;
%     end
%     rects(:,3:4) = rects(:,3:4) + rects(:,1:2);
%     rects = clip_to_image(rects,I);
%     A = computeHeatMap(I,rects,'max');
%     sc(cat(3,A,I),'prob'); pause
%     
%     
    
    %     clf; imagesc(I); axis image; pause;
end
% end