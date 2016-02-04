% Experiment 0039 %
% June 5'th, 2014

%% The purpose of this experiment is to - predict locations + extents of action objects.

%% initialization
default_init;
specific_classes_init; % class_labels, isTrain
% newImageData = newImageData(validIndices);

% dataInd = findImageIndex(data,curImageData.imageID);
load ~/storage/misc/upper_bodies.mat



%%
nFeat = 0;
%% step 1: assume that faces have been given manually and try to predict locations of action objects.
%% get locations of face + object bounding boxes for all images.
conf.get_full_image = true;
n = length(validIndices);
face_boxes = zeros(n,4);
obj_boxes = zeros(n,4);
person_boxes = zeros(n,4);
debug_ = false;
validFaces = true(size(validIndices));
rPath = '~/storage/misc/action_pred_data.mat';
if (exist(rPath,'file'))
    load(rPath);
else    
    % validIndices = validIndices(1:100);
    for t = 1:length(validIndices)
        t
        k = validIndices(t);
        curImageData = newImageData(k);
        needImage = false;
        dataInd = findImageIndex(data,curImageData.imageID);
        L = load(j2m('~/storage/s40_upper_body_faces/',curImageData.imageID));
        if (~isempty(L.res)) % found a face...
            curImageData.alternative_face = L.res(1,1:4)+data(dataInd).upperBodies([1 2 1 2]);
        else
            validFaces(t) = false;
        end
        
        [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = ...
            getSubImage(conf,curImageData,1,false,use_manual_faces && validFaces(t),needImage);
        face_boxes(t,:) = face_box;
        person_boxes(t,:) = curImageData.I_rect;
        obj_boxes(t,:) = curImageData.obj_bbox(1:4);
        
        % get the saliency image...
        sal = foregroundSaliency(conf,curImageData.imageID);
        
        if (debug_)
            I = getImage(conf,curImageData);
            clf; subplot(1,2,1);imagesc2(I); hold on;
            plotBoxes(face_box);
            plotBoxes(curImageData.I_rect,'mo','MarkerFaceColor','m');
            plotBoxes(curImageData.obj_bbox,'g--','LineWidth',2);
            
            subplot(1,2,2);imagesc2(sal); hold on;
            plotBoxes(face_box);
            plotBoxes(curImageData.I_rect,'mo','MarkerFaceColor','m');
            plotBoxes(curImageData.obj_bbox,'g--','LineWidth',2);
            drawnow;pause
        end
    end
    save(rPath,'face_boxes','obj_boxes');
end
[face_h,face_w,face_a] = BoxSize(face_boxes);
[obj_h,obj_w,obj_a] = BoxSize(obj_boxes);
img_rects = cat(1,newImageData(validIndices).I_rect);
[rect_h,rect_w,rect_a] = BoxSize(img_rects);



% [poseletPreds,allProbs] = learnBoxPredictions(conf,action_rois);
%%
outPath = '~/storage/s40_action_pred';
ensuredir(outPath);
debug_ = false;override = true;
for iClass = 1:length(unique(class_labels))
    
    sel_ = class_labels ==iClass;
    sel_train = sel_ & isTrain;
    % try to predict location and size of object boxes from face boxes.
    f = 15.0./face_h;
    
    face_centers = boxCenters(face_boxes);
    obj_centers = boxCenters(obj_boxes);
    rect_centers = boxCenters(img_rects);
    sel_test = sel_ & ~isTrain;
    xy = (obj_centers-face_centers).*[f f];
    rads = f.*((obj_a/pi).^.5);
    plotPolygons(xy(sel_,:),'r+');axis image
    D = 150;
    Z = zeros(D);
    for t=1:length(sel_)
        if (sel_train(t))
            %         t
            curPoly = circleToPolygon([D/2+xy(t,:) rads(t)]);
            Z = Z + poly2mask2(curPoly,size(Z));
        end
    end
    Z = Z/nnz(sel_train);
    Z = (Z+fliplr(Z))/2;
    if (debug_)
        figure,imagesc2(Z)
        hold on; plotPolygons(circleToPolygon([[D D]/2 5]),'g','LineWidth',3);
    end
    
    
    
    % %
    
    % try this on a specific image.
    %     ir = find(sel_test);
    %     for t = 1:length(ir)
    for t = 1:length(validIndices)
        %         if (~sel_test(t))
        %             continue;
        %         end
        t
        k = validIndices(t);
        %         break
        %k = ir(t);
        
        resPath = j2m(outPath, newImageData(k).imageID);
        
        if (~debug_ && ~override && exist(resPath,'file'))
            continue;
        end
        
        I = getImage(conf,newImageData(k));
        
        f1 = 100/size(I,1);
        I = imResample(I,f1,'bilinear');
        % imshow(I)
        % rescale Z and translate it so the face is the center...
        Z1 = imResample(Z,f1*face_h(t)/10);
        Z1 = flipud(Z1);
        ff = fliplr(round(face_centers(t,:)*f1));
        Q = zeros(size2(I)); Q(ff(1),ff(2)) = 1;
        
        % A = zeros(407);
        % A(203,203) = 1;
        % A = imfilter
        % A = imfilter(A,Z1);
        Q = imfilter(Q,Z1,'corr');
        Q = Q/max(Q(:));
        if (debug_)
            clf; imagesc2(sc(cat(3,Q,im2double(I)),'prob'));
            %             close all;
            pause;
        else
            save(resPath,'Q');
        end
        
        
    end
    %     break
    % figure,imagesc2(Q);
    % figure,imagesc2(I);
end
%% take 2
outPath = '~/storage/s40_action_pred';
ensuredir(outPath);
debug_ = false;override = true;
for iClass = 1:length(unique(class_labels))
    
    sel_ = class_labels ==iClass;
    sel_train = sel_ & isTrain;
    % try to predict location and size of object boxes from face boxes.    
    
    % predict locations of objects using face rectangles and person
    % rectangles.           
    face_centers = boxCenters(face_boxes);
    obj_centers = boxCenters(obj_boxes);
    rect_centers = boxCenters(img_rects);    
    diff_face=(obj_centers-face_centers)./[face_w face_h];
    s_face = (obj_a./face_a).^.5;  
    diff_rect=(obj_centers-rect_centers)./[rect_w rect_h];
    s_obj = (obj_a./rect_a).^.5;                           
    sel_test = sel_ & ~isTrain;    
    for t = 1:length(validIndices)
%         t = 2258
                if (~sel_test(t))
                    continue;
                end
        t
        k = validIndices(t);
        %         break
        %k = ir(t);
        
        resPath = j2m(outPath, newImageData(k).imageID);
        
        if (~debug_ && ~override && exist(resPath,'file'))
            continue;
        end
        
        [I,I_rect] = getImage(conf,newImageData(k));
        clf;subplot(2,2,1);imagesc2(I);
        pred_rect = np_predict(I,I_rect,diff_rect(sel_train,:));        
        pred_face = np_predict(I,face_boxes(t,:),diff_face(sel_train,:));
        pred_face_all = np_predict(I,face_boxes(t,:),diff_face);        
        pred_rect_all = np_predict(I,face_boxes(t,:),diff_rect);
        pred_face = normalise(pred_face);
        pred_rect = normalise(pred_rect);
        subplot(2,2,2);imagesc2((pred_rect+pred_face)); title('pred rect');
        
        
        sal = foregroundSaliency(conf,newImageData(k).imageID);
        subplot(2,2,3);imagesc2(sal);title('sal');
        
        subplot(2,2,4); imagesc2(sc(cat(3,(pred_rect+pred_face)/2,I),'prob'));        
        
        pause;continue;
        f1 = 100/size(I,1);
        I = imResample(I,f1,'bilinear');
        % imshow(I)
        % rescale Z and translate it so the face is the center...
        Z1 = imResample(Z,f1*face_h(t)/10);
        Z1 = flipud(Z1);
        ff = fliplr(round(face_centers(t,:)*f1));
        Q = zeros(size2(I)); Q(ff(1),ff(2)) = 1;
        
        % A = zeros(407);
        % A(203,203) = 1;
        % A = imfilter
        % A = imfilter(A,Z1);
        Q = imfilter(Q,Z1,'corr');
        Q = Q/max(Q(:));
        if (debug_)
            clf; imagesc2(sc(cat(3,Q,im2double(I)),'prob'));
            %             close all;
            pause;
        else
            save(resPath,'Q');
        end
        
        
    end
    %     break
    % figure,imagesc2(Q);
    % figure,imagesc2(I);
end

