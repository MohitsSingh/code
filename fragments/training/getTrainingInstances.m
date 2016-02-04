function samples = getTrainingInstances(globalOpts,image_ids,model,cls)
% find nTrain examples of for each class in the training images.
% extract those samples

% for each object class, enter as positives the ground truth bounding
% boxes. enter as negatives the boxes extracted from the current image
% which have 20-50 overlap with the ground truth bounding box.

% TODO :at a later stage, use the classifier to mine hard negatives.

% trainInstances = struct('image_id',{},'label',{},'bbox',{},'hist',{},...
%     'size_ratio',{},'labelIDX',{});
% sampleCount = 0;

% for each class, find nTrain boxes which conincide nicely with the ground
% truth boundingboxes and label them as positive samples.

class_subset = globalOpts.class_subset;
% areas = getBboxArea(bboxes).^.5;
minOverlap = .5;

%%
samples = struct('classID',{},'posFeatureVecs',{},'negFeatureVecs',{});

classID = globalOpts.VOCopts.classes{cls};
samples(1).classID = classID;
[ids,t] = textread(sprintf(globalOpts.VOCopts.imgsetpath,[classID '_train']),'%s %d');
ids = ids(t==1);
sample_ids = ids;
posSampleIDS = [];
negSampleIDS = [];
posBoxes = [];
negBoxes = [];

for iID = 1:length(sample_ids)
    currentID = ids{iID};
    % get non-difficult object examples of current class for this
    % image.
    rec = PASreadrecord(getRecFile(globalOpts,currentID));
    clsinds=strmatch(classID,{rec.objects(:).class},'exact');
    isDifficult = [rec.objects(clsinds).difficult];
    isTruncated = false(size([rec.objects(clsinds).truncated]));%this effectively
    % toggles usage of truncated examples. 'false' means they are used.
    % get bounding boxes
    gtBoxes = cat(1,rec.objects(clsinds(~isDifficult & ~isTruncated)).bbox);
    difficultBoxes = cat(1,rec.objects(clsinds(isDifficult)).bbox);
    % convert to ymin xmin ymax xmax format
    
    if (isempty(gtBoxes))
        continue;
    end
    gtBoxes = gtBoxes(:,[2 1 4 3]);
    nGTBoxes = size(gtBoxes,1);
    % get segmentation boxes for current image
    L = load(getBoxesFile(globalOpts,currentID));
    imageBoxes = L.boxes;
    % remove 'bad' boxes
    imageBoxes = imageBoxes(~bad_bboxes(imageBoxes,globalOpts),:);
    
    if (~globalOpts.partial_det) % find whole objects
        
        overlaps = boxesOverlap(imageBoxes,gtBoxes);
        
        % find also boxes from segmentation that overlap well with ground-truth boxes
        pos_inds = sum(overlaps >= 9.9,2) > 0;  %TODO - this adds NOTHING
        gtBoxes = [gtBoxes;imageBoxes(pos_inds,:)];
                
        % of course, make sure that the negative examples
        % don't by accident overlap another positive example with
        % more than .5....
        neg_inds = sum(overlaps >= .2,2) > 0 &...
            sum(overlaps < .5,2) == nGTBoxes;
        
        % NOW update nGTboxes (doing so before would have been wrong).
        nGTBoxes = size(gtBoxes,1);
        
                        
        % be careful not to retain as negatives those that overlap with
        % any difficult examples.
        if (~isempty(difficultBoxes))
            difficultBoxes = difficultBoxes(:,[2 1 4 3]);
            overlaps_diff = boxesOverlap(imageBoxes,difficultBoxes);
            neg_inds = neg_inds & (sum(overlaps_diff >= .5) == 0);
        end
        
        % remove overlapping negatives
        imageBoxes = imageBoxes(neg_inds,:);
        keep = boxes_removeNearDup(imageBoxes);
        imageBoxes = imageBoxes(keep,:);
        nNegBoxes = size(imageBoxes,1);
        posSampleIDS = [posSampleIDS;repmat({currentID},nGTBoxes,1)];
        posBoxes = [posBoxes;gtBoxes];
        
        negSampleIDS = [negSampleIDS;repmat({currentID},nNegBoxes,1)];
        negBoxes= [negBoxes;imageBoxes];
    else
        
        % relative area of bboxes occupied by ground truth (most of it
        % should be covered)
        gt_in_boxes = overlap_assym(gtBoxes,imageBoxes); 
        % relative area of ground truth covered by bboxes (a nontrivial
        % portion should be covered);
        boxes_in_gt = overlap_assym(imageBoxes,gtBoxes)';         
   
        % require >= 20% of any gt and 80% of bbox inside gt
        qq_pos = sum(gt_in_boxes > .8 & boxes_in_gt > .2,1) > 0;
        
        % require less than 20% of gt occupied for all ground truth boxes
        % and that less that 20% of box is in gt...
        qq_neg = sum(boxes_in_gt > .2,1) == 0 & ...
            sum(gt_in_boxes > .2,1)==0;
        
        assert(sum(qq_pos & qq_neg) == 0);
        cur_pos = imageBoxes(qq_pos,:);
        cur_neg = imageBoxes(qq_neg,:);
%         cur_neg = cur_neg(vl_colsubset(1:size(cur_neg,1), size(cur_pos,1)),:);
        
        cur_pos = cur_pos(boxes_removeNearDup(cur_pos),:);
        cur_neg = cur_neg(boxes_removeNearDup(cur_neg),:);
        posBoxes = [posBoxes;cur_pos];
        negBoxes = [negBoxes;cur_neg];                
        posSampleIDS = [posSampleIDS;repmat({currentID},size(cur_pos,1),1)];
        negSampleIDS = [negSampleIDS;repmat({currentID},size(cur_neg,1),1)];
        
%         max(overlaps_box_truth) < .2
%         find as positives those that overlap nicely with aeroplanes...                
%         for k = 1:length(imageBoxes)
%             if (qq_neg(k) > 0)
%                 clf;
%                 imshow(getImageFile(globalOpts,currentID));
%                 hold on;
%                 plotBoxes2(imageBoxes(k,:),'Color','r','LineWidth',2);
%                 plotBoxes2(gtBoxes,'Color','g','LineWidth',2);
%                 pause
%             end
%         end
    end        
end

pos_feats = [];
neg_feats = [];
for scales = 1
    goods = ~bad_bboxes(posBoxes*scales,globalOpts);    
    f = find(goods);
%     f = f(1:10);
    pos_feats = [pos_feats,get_box_features2(globalOpts,posBoxes(f,:),posSampleIDS(f),model,scales)];           
    goods = ~bad_bboxes(negBoxes*scales,globalOpts);
%     profile on;

    f = find(goods);
%     f = f(1:10);
    neg_feats = [neg_feats,get_box_features2(globalOpts,negBoxes(f,:),negSampleIDS(f),model,scales)];
%     profile viewer;
end

samples.posFeatureVecs = pos_feats;
samples.negFeatureVecs = neg_feats;

function keep = boxes_removeNearDup(boxes)
olp = boxesOverlap(boxes);
[ii,jj] = find(olp >= .7);
keep = true(size(boxes,1),1);
for q = 1:length(ii)
    % check if source not already removed
    % if so, keep current...
    if (keep(ii(q)))
        keep(jj(q)) = false;
    end
end

% area (bboxes2.int.bboxes) / area (bboxes2)
function overlaps = overlap_assym(bboxes,bboxes2)
    nBoxes = size(bboxes,1);
    nBoxes2 = size(bboxes2,1);
    overlaps = zeros(nBoxes,nBoxes2);
    
    for iBox = 1:nBoxes2
        q = repmat(bboxes2(iBox,:),nBoxes,1);
        boxes_i = BoxIntersection(q,bboxes);
        has_intersection = boxes_i(:,1) ~= -1;
        [~, ~, bi] = BoxSize(boxes_i(has_intersection,:));
        [~,~,b2] = BoxSize(q(has_intersection,:));
        [~,~,b1] = BoxSize(bboxes(has_intersection,:));
        overlaps(has_intersection,iBox) = bi./b2;
    end
