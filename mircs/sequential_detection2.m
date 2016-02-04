
% imshow(multiImage((train_faces(t_train))))

if (~exist('toStart','var'))
    initpath;
    config;
    %%
    load lipData.mat;
    
    initLandMarkData;
   
    
    
    % root: detect face at given pose.
    % node 1: detect lips / not detect lips.
    % node 2: detect stra
    % w / not detect straw.
    
    % root is given, now proceed to detect lips. use lips from faces with good
    % scores.
    
    face_comp_train = [faceLandmarks_train_t.c];
    face_comp_test = [faceLandmarks_test_t.c];
    % stage 1: assume lips are detected correctly, find straight line segments
    % from lips.
            
    lipImages_test_2 = multiCrop(conf,lipImages_test,[], [100 NaN]);
    lipImages_train_2 = multiCrop(conf,lipImages_train,[], [100 NaN]);
    
    clusters = train_patch_classifier(conf,[],getNonPersonIds(VOCopts),'suffix','multilips',...
        'override',false);
    
    % learn the appearance of cups....
    conf.class_subset = conf.class_enum.DRINKING;
    % roiPath = '~/storage/action_rois';
    %     [action_rois,true_ids] = markActionROI(conf,roiPath);
    
    L = load(fullfile('~/storage/gbp_train',sprintf('%05.0f.mat',k)));
    
    
    
    
    check_test = false;
    clear test_faces_2;
    clear get_full_image;
    if (check_test)
        suffix = 'test';
        cur_t = t_test;
        cur_set = lipImages_test;
        cur_dir = '~/storage/gbp_test/';
        cur_comp = face_comp_test;
        cur_face_scores = test_faces_scores_r;
        imageIDs = test_ids_r;
        curLipBoxes = lipBoxes_test_r_2;
        curFaces = test_faces;
        locs_r = test_locs_r;
    else
        cur_t = t_train
        cur_dir = '~/storage/gbp_train/'
        cur_set = lipImages_train;
        suffix = 'train';
        cur_comp = face_comp_train;
        cur_face_scores = train_faces_scores_r;
        imageIDs = train_ids_r;
        curLipBoxes = lipBoxes_train_r_2;
        curFaces = train_faces;
        locs_r = train_locs_r;
    end
    
    clear qq_test;
    clear qq_train;
    clear lipImages_test_2;
    clear lipImages_train_2;
    clear lipImages_test;
    clear lipImages_train;
    clear ucm;
    toStart = 0;
end
f = find(cur_t);
%r_sel = f([[2 5 9 14 20 21 23 28 29 41 44 45]]);
r_sel = f([1 5 6 14 18 19 21 23 27 31 38 42 46 51 54]);
% r_sel = f([19 21 23 27 31 38 42 46 51 54]);
addpath(genpath('/home/amirro/code/3rdparty/proposals/src/GeometricContext/'));

for k =413:length(cur_t) % woman with straw
        close all;
    k
    imageInd = k;
    
    if (~ismember(k,r_sel))
%         continue;
    end
    
%     if (checkedImages(imageInd))
%         continue;
%     end
    
    if(~cur_t(imageInd))
        continue;
    end
    
    curLipRect = curLipBoxes(imageInd,:);
    faceImage = curFaces{imageInd};

    curTitle = '';
    curFaceBox = locs_r(imageInd,:);
    
    if (curFaceBox(:,conf.consts.FLIP))
        curTitle = 'flip';
        curLipRect = flip_box(curLipRect,[128 128]);        
%         ori = -ori;
    end
    
    conf.not_crop = false;
    [fullImage,xmin,xmax,ymin,ymax] = getImage(conf,imageIDs{imageInd});
    
   conf.not_crop = 1;
    [f2,xmin,xmax,ymin,ymax] = getImage(conf,imageIDs{imageInd});
    curBoxShifted = curFaceBox(1:4)+[xmin ymin xmin ymin];
    imshow(f2); hold on;plotBoxes2(curBoxShifted([2 1 4 3]));
    
%     lineseg
    
%     faceIm = im2double(cropper(f2,round(inflatebbox( curBoxShifted, [1 2], 'post', false))));
%     faceIm = im2double(train_faces{k});
%     imshow(fullImage) 
%     G = imresize(rgb2gray(faceIm),2);
%     [lines,edgeIM] = APPgetLargeConnectedEdges(G,5); 
%     clf,imshow(edgeIM)
%     pause;
%     continue;
    
    conf.not_crop = false;
    
    gpbFile = fullfile('/home/amirro/storage/gpb_s40/',strrep(imageIDs{imageInd},'.jpg','.mat'));
    k
    if (~exist(gpbFile,'file'))
        fprintf(2,'warning : gpb file does not exist for %s\n',imageIDs{imageInd});
        break;
    end
    
%     hold on;
        
    
%     plotBoxes2(locs_r(imageInd,[2 1 4 3]));
    % shift the lip detection...
    topleft_ = locs_r(imageInd,[1:2]);
    [rows cols ~] = BoxSize(locs_r(imageInd,:));
    lipRectShifted = rows*curLipRect/128 + [topleft_ topleft_];
    [rows2 cols2 ~] = BoxSize(lipRectShifted);
%     plotBoxes2(lipRectShifted(:,[2 1 4 3]));            
       
%     title(curTitle);
    L = load(gpbFile);
    gPb_orient = double(L.gPb_orient);
    ucmFile = strrep(gpbFile,'.mat','_ucm.mat');
    if (exist(ucmFile,'file'))
        load(ucmFile);
        curUCM = ucm;
    else
        curUCM = contours2ucm(gPb_orient);
    end
                            
    % make a graph out of the ucm.
%     regionTree = makeRegionTree(curUCM);
    
    regionsFile = strrep(gpbFile,'.mat','_regions.mat');
    if (exist(regionsFile,'file'))
        fprintf('regions file exists! woohoo!\n');
        load(regionsFile);
    else
        regions  = combine_regions(curUCM,.5);
        regionOvp = regionsOverlap(regions);
    end
        
    edgeIm = curUCM >= .1;
    
    
    
    
            

    %     figure,imagesc(curUCM.*(L.gPb_thin>0))
    
    T_ovp = 1; % don't show overlapping regions...
%     regions = suppresRegions(regions, regionOvp, T_ovp); % help! we're being oppressed! :-)
%     close all;
%     [~,bb2] = imcrop(f2);
%     [ovps,ints,areas] = boxRegionOverlap(imrect2rect(bb2),regions,dsize(f2,1:2));
%     scores = ints./areas;
%     scores = scores.*(scores > .7);
%     [o,io] = sort(scores,'descend');
%     io = io(o>0);
%     close all    
%     displayRegions(f2,regions,io,0); % just find elongated elements
%             
%         continue;
    origBoundaries = ucm<=.1;
    segLabels = bwlabel(origBoundaries);
    %     figure,imagesc(segLabels)
    segLabels = imdilate(segLabels,[1 1 0]);
    segLabels = imdilate(segLabels,[1 1 0]');
    
    S = medfilt2(segLabels); % close the small holes...
    
    segLabels(segLabels==0) = S(segLabels==0);
    assert(isempty(find(segLabels==0)));
    figure(1); clf;
    h = zeros(3,1);
    h(1) = subplot(1,3,1);
    imshow(f2); hold on;
    
    plotBoxes2(curBoxShifted([2 1 4 3]));
    h(2) = subplot(1,3,2);
    imshow(curUCM); hold on;    
    lipRect = lipRectShifted + [xmin ymin xmin ymin];
    plotBoxes2(lipRect([2 1 4 3]),'g');colormap jet; 
    [M,O] = gradientMag(im2single(f2));
    h(3)=subplot(1,3,3); 
    imshow(M,[]);colormap jet; 
    linkaxes(h,'xy');
    
    %%
     Z = curUCM >.15; % edges.
     D_edge = bwdist(Z);
     imagesc(Z)
     B = bwmorph(~Z,'skel',inf); % "middle" of segments.
%      figure,imshow(B);    
     pixelCost = double(B.*D_edge);     
%      figure,imagesc( exp(-pixelCost).*B);
     
     %
     pixelCost(~B) = max(pixelCost(B));
     pixelCost(D_edge > 10) = 0;
     
%      figure,imagesc( pixelCost)
     % % 
%      figure,imagesc(Z)
     
     % make a pixel graph.                            
     [X,Y] = meshgrid(1:size(Z,2),1:size(Z,1));
     inds = sub2ind(size(Z),Y,X);
     m = numel(inds);
     M = sparse(m,m);
     sz = size(Z);
     
     % discard all border indices for simplicity.     
     inds_right = sub2ind(sz,Y(2:end-1,2:end-1),X(2:end-1,2:end-1)+1);
     inds_bottom = sub2ind(sz,Y(2:end-1,2:end-1)+1,X(2:end-1,2:end-1));
     inds_bottomright = sub2ind(sz,Y(2:end-1,2:end-1)+1,X(2:end-1,2:end-1)+1);
     inds_bottomleft = sub2ind(sz,Y(2:end-1,2:end-1)+1,X(2:end-1,2:end-1)-1);
     inds_ = inds(2:end-1,2:end-1);
     
     transition_side = 1;
     transition_diag = sqrt(2);
     
     costs_right = min(pixelCost(inds_),pixelCost(inds_right))+transition_side;
     costs_bottom = min(pixelCost(inds_),pixelCost(inds_bottom))+transition_side;
     costs_bottomright = min(pixelCost(inds_),pixelCost(inds_bottomright))+transition_diag;
     costs_bottomleft = min(pixelCost(inds_),pixelCost(inds_bottomleft))+transition_diag;
     M = max(M,sparse(inds_,inds_right,costs_right,m,m));
     M = max(M,sparse(inds_,inds_bottom,costs_bottom,m,m));
     M = max(M,sparse(inds_,inds_bottomright,costs_bottomright,m,m));
     M = max(M,sparse(inds_,inds_bottomleft,costs_bottomleft,m,m));
     M = max(M,M');
     
     [Y_,X_] = ind2sub(size(Z),inds_);
%      imagesc(pixelCost);
     
%      gplot(M,[X(:),Y(:)]);

    figure(1); clf; imshow(f2); [x,y] = ginput(1);
    sourceV = inds(round(y),round(x));

%     figure(1),imshow(f2)
%    [D,P] = dijkstra( G, sourceV );
 
    [dist, path, pred] = graphshortestpath(M,sourceV,'Directed',false);
    
    shortestPathImage = reshape(dist,size(Z));
    figure(1);clf(1);imagesc(exp(-shortestPathImage/1000));
    figure(2);clf(2);,imagesc(f2);
    
%     figure,imagesc(pixelCost)
    
%     clf,figure(1);hist(exp(-shortestPathImage(:)/10000));
    
    
    %%
    pause;            
    continue;
    Z = false(dsize(f2,1:2));
    bc = round(boxCenters(lipRect));
    Z(bc(2),bc(1)) = 1;
    Z = imdilate(Z,ones(30));
    ovp = boxRegionOverlap(Z,regions,dsize(f2,1:2));     
%     ovp = ovp(ovp>0);
    regions  = regions(ovp>0);
    ovp = ovp(ovp>0);
    regions = fillRegionGaps(regions);
    allProps ={};
    for ii=1:length(regions) 
        allProps{ii} = regionprops(regions{ii},'Eccentricity','MajorAxisLength','MinorAxisLength');
    end
    allProps = [allProps{:}];
    
    scores = [allProps.Eccentricity].*([allProps.MinorAxisLength] < 15);
    [o,io] = sort(ovp+scores,'descend');
    io = io(o > 0);   
    figure(2);displayRegions(f2,regions,io(1:min(length(io),5)));
    
   
       
end
