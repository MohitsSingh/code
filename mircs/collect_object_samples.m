
function [initialSamples,all_ims] = collect_object_samples(conf,dataset,curClass)
conf.clustering.sample_method = 'none';
conf.clustering.min_hog_energy = 0;
conf.detection.params.detect_min_scale = .01;
conf.detection.params.detect_max_scale = 1;
images = dataset_list(dataset,'train',curClass);
all_x = {};
all_ims = {};
all_boxes = {};
image_paths = {};
progress;
% images = vl_colsubset(images,100);
for k = 1:length(images)
        k
    progress(k,length(images))
    %     fileName = sprintf('tmp_10/%03.0f',k);
    %     if (exist(fileName,'file'))
    %         continue;
    %     end
    
    [bbox,im]=  dataset_image(dataset,curClass,images{k});
    
     % limit the box to the top, to get samples from the top part.
    %bbox(:,4) = bbox(:,4)/2;
    
    % just one bbox from each image
%     bbox = bbox(1,:);
        
    bbox(:,3:4) = bbox(:,3:4)+bbox(:,1:2);
    % sample random features around the bbox.
    
    bbox = round(inflatebbox(bbox,[1.3,1.3],'both',false));
    bbox = clip_to_image(round(bbox),im);
    
    for q = 1:size(bbox,1)
        
        
        curIm = cropper(im,bbox(q,:));
        curIm = rescaleImage(curIm,180);
        [X,uus,vvs,scales,t,boxes ] = allFeatures( conf,curIm, .5 );
% %         norms = sum(X.^2,1).^.5;
% %         X(:,norms < 2) = [];
% %         boxes((norms < 2,:) = [];
        %norms = sum(cat(2,all_x{:}).^2,1).^.5;
        if (isempty(X))
            continue;
        end
        % sample 4 boxes from each image: top half; top quarter
        imgBox = [1 1 dsize(curIm,[2 1])];
        h = imgBox(4);
        
        
        
        box1 = [1 1 imgBox(3) h/2];
        box2 = [1 1 imgBox(3) h/4];
%         box3 = [1 1 imgBox(3) h/8];
        box4 = imgBox;
        
        targetBoxes = [box1;box2;box4];
        ovp = boxesOverlap(boxes,targetBoxes);
        [m,im] = max(ovp,[],1);
        
        im = unique(im);
                
        boxes = boxes(im,1:4);
        X = X(:,im);
% % %         
% % %                         
% % %         
% % %         
% % %         %r = randperm(size(X,2));
% % %         % prefer low scales
% % %         r = size(boxes,1):-1:1;
% % %         boxes = boxes(r,[1:4 end]);
% % %         X = X(:,r);
% % %         pick = nms(boxes,.5);
% % %         boxes=boxes(pick,:);
% % %         X = X(:,pick);
        all_ims = [all_ims;col(multiCrop(conf,{curIm},round(boxes)))];
        all_x = [all_x,{X}];
%         if (~isempty(boxes))
%             clf,imagesc(curIm); axis image; hold on; plotBoxes(boxes,'g');
%             plotBoxes(targetBoxes,'r');
%             pause;
%         end
    end
    
    %     fclose(fopen(sprintf('tmp_10/%03.0f',k),'w'));
end

%save bottle_patches.mat all_ims all_x
initialSamples = cat(2,all_x{:});
% all_x = cat(2,all_x{:});


% for k = 1:length(ims)
%     clf; imagesc(ims{k}); axis image; pause;
% end


function [im,scaleFactor] = rescaleImage(im,maxHeight)
if (size(im,1) > maxHeight)
    scaleFactor = maxHeight/size(im,1);
else
    scaleFactor = 1;
end
im = imResample(im,scaleFactor,'bilinear');

% % % function [initialSamples,all_ims] = collect_object_samples(conf,dataset,curClass)
% % % conf.clustering.sample_method = 'none';
% % % conf.clustering.min_hog_energy = 0;
% % % images = dataset_list(dataset,'train',curClass);
% % % all_x = {};
% % % all_ims = {};
% % % all_boxes = {};
% % % image_paths = {};
% % % progress;
% % % images = vl_colsubset(images,100);
% % % for k = 1:length(images)
% % %         k
% % %     progress(k,length(images))
% % %     [bbox,im]=  dataset_image(dataset,curClass,images{k});
% % %
% % %     % just one bbox from each image
% % %     %     bbox = bbox(1,:);
% % %
% % %     bbox(:,3:4) = bbox(:,3:4)+bbox(:,1:2);
% % %     [im,scaleFactor] = rescaleImage(im,360); % make sure largest dimension is not too large.
% % %     % sample random features around the bbox.
% % %     bbox = bbox*scaleFactor;
% % %
% % %     % sample features with an overlap of at most .5 (per each scale)
% % %     conf.detection.params.detect_min_scale = 1;
% % %     [X,uus,vvs,scales,t,boxes ] = allFeatures( conf,im,.5);
% % %     if (isempty(boxes))
% % %         continue;
% % %     end
% % %
% % %     intersection = BoxIntersection2(bbox,boxes);
% % %     [~,~,areas] = BoxSize(boxes);
% % %     areas = areas';
% % %     intersection = bsxfun(@rdivide,intersection,areas);
% % %
% % %     sel_ = max(intersection,[],1) > .5;
% % %
% % %     boxes = boxes(sel_,:);
% % %     X = X(:,sel_);
% % %
% % %     inds = randperm(size(boxes,1));
% % %     boxes = boxes(inds,:);
% % %     X = X(:,inds);    %
% % %     pick = nms(boxes,.5);
% % %     boxes=boxes(pick,:);
% % %     X = X(:,pick);
% % %
% % %     all_ims = [all_ims;col(multiCrop(conf,{im},round(boxes(:,1:4))))];
% % %     all_x = [all_x,{X}];
% % % %     if (~isempty(boxes))
% % % %         clf,imagesc(im); axis image; hold on; plotBoxes(boxes,'g');
% % % %         pause;
% % % %     end
% % % end
% % %
% % % %     fclose(fopen(sprintf('tmp_10/%03.0f',k),'w'));
% % % % end
% % %
% % % %save bottle_patches.mat all_ims all_x
% % % initialSamples = cat(2,all_x{:});
% all_x = cat(2,all_x{:});


% for k = 1:length(ims)
%     clf; imagesc(ims{k}); axis image; pause;
% end


% % % older version
