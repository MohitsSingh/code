function curFeats = extractCandidateFeatures3(conf,imageSet,salData,k,...
    clip_to_mouth,debug_)
%Rs = extractCandidateFeatures3(conf,imageSet,salData,k)
if (nargin < 5)
    debug_ = false;
end
curFeats = struct('bbox',{},'bc',{},'horz_extent',{},'y_top',{},'ucmStrength',{},'isConvex',{},...
    'salStrength',{},'candidates',{});
[I,mouth_mask,subUCM] = init_();

if (isempty(subUCM))
    return;
end

% salImage = salData{k};
% salImage = imresize(salImage,size(subUCM),'bilinear');
ucmORIG = subUCM;

% generate_u_shapes();

% edgeCandidtates = generate_edge_candidates();

edgeImages = {};
subUCMS = {};

% for T = [.3 .6 max(ucmORIG(:))]
%     T
%     E = ucmORIG.*(ucmORIG>=T);
%     if (nnz(E) < 10)
%         continue;
%     end
%     edgeImages{end+1} = E;
%     subUCMS{end+1} = E;
% end
edgeImages{end+1} = edge(rgb2gray(im2double(I)),'canny');
subUCMS{end+1} = ucmORIG;
for kk = 1:length(edgeImages)
    E = edgeImages{kk};
    [seglist,edgelist] = processEdges(E);
    
    % remove components not "touching" the mouth region.
    toRemove = false(size(edgelist));
    if (clip_to_mouth)
        for k = 1:length(edgelist)
            yx = edgelist{k};
            inds = sub2ind2(size(E),yx);
            if (~any(mouth_mask(inds)))
                toRemove(k) = true;
            end
        end
    end
    
    edgelist(toRemove) = [];
    candidates = findConvexArcs2(seglist,E,edgelist);
    if (isempty(candidates))
        continue;
    end
    subUCM = subUCMS{kk};
    curFeats = [curFeats;getFeatures(candidates)];
end
    function [I,mouth_mask,subUCM] = init_()
        currentID = imageSet.imageIDs{k};
        [I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
        ucmFile = fullfile('/home/amirro/storage/gpb_s40/',strrep(currentID,'.jpg','_ucm.mat'));
        load(ucmFile); % ucm
        ucm = ucm(ymin:ymax,xmin:xmax); %#ok<NODEF>
        bbox = round(imageSet.faceBoxes(k,1:4));
        bbox = clip_to_image(bbox,I);
        mouthBox = round(imageSet.lipBoxes(k,1:4)-bbox([1 2 1 2]));
        mouthBox = clip_to_image(mouthBox,I);
        
        subUCM = ucm(bbox(2):bbox(4),bbox(1):bbox(3));
        mouth_mask = false(size(subUCM));
        mouth_mask(mouthBox(2):mouthBox(4),mouthBox(1):mouthBox(3)) = true;
        I = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
    end
    function curFeats = getFeatures(candidates)
        if (isempty(candidates))
            return;
        end
        
        candidates = fixSegLists(candidates);
        % get the following features:
        % center, width, height, color, saliency
        
        ims = cellfun2(@(x)(paintLines(zeros(size(E)),x)>0),candidates);
        ff = cellfun2(@row,ims);
        ff = cat(1,ff{:});
        [ff,ia] = unique(ff,'rows');
        candidates = candidates(ia);
        ims = ims(ia);
        bboxes = zeros(length(candidates),4);
        y_tops = zeros(length(candidates),1);
        zz = [length(candidates),1];
        ucmStrengths = zeros(zz);
        salStrengths = zeros(zz);
        isConvex = false(zz);
        intersectMouth = false(zz);
        ims_show = cellfun2(@(x).7*rgb2gray(I)+...
            (paintLines(zeros(size(E)),x)>0),candidates);
        % calculate angle from top of image, check convexity
        topCenter = [size(E,2)/2,1];
        
        
        for iCandidate = 1:length(candidates)
            %         iCandidate
            pts = candidates{iCandidate};
            pts = [pts(1:end,1:2);pts(end,3:4)];
            
            % intersects with mouth?
            intersectMouth(iCandidate) = any(mouth_mask(sub2ind2(size(E),pts)));
            
            pts = fliplr(pts); % x,y
            d = bsxfun(@minus,pts,topCenter);
            tg = atan2(d(:,2),d(:,1));
            x = pts(:,1);
            y = pts(:,2);
            [s,is] = sort(tg,'ascend');
            if (is(1) > is(end))
                inc = -1;
            else
                inc = 1;
            end
            x = x(is(1):inc:is(end));
            y = y(is(1):inc:is(end));
            
            pts = [x y];
            diffs = diff(pts);
            crosses = zeros(size(diffs,1)-1,1);
            for id = 1:length(crosses)
                crosses(id) = diffs(id,2)*diffs(id+1,1)-diffs(id,1)*diffs(id+1,2);
            end
            
            isConvex(iCandidate) = ~any(crosses<0);
        end
        
        %%%%%%%%
        % ims = ims(isConvex);
        % candidates = candidates(isConvex);
        
        for iCandidate = 1:length(candidates)
            lineImage = ims{iCandidate};
            [y,x] = find(lineImage);
            bboxes(iCandidate,:) = pts2Box([x y]);
            %         chull = convhull(x,y);
            %ucmStrengths(iCandidate) = mean(subUCM(imdilate(lineImage,ones(3)) & subUCM > 0));
            %ucmStrengths(iCandidate) = sum(subUCM(imdilate(lineImage,ones(3)) & (subUCM > 0)))/nnz(lineImage);
            ucmStrengths(iCandidate) = mean(subUCM(imdilate(lineImage,ones(3)) & (subUCM > 0)));
            %             salStrengths(iCandidate) = mean(salImage(imdilate(lineImage,ones(3))));
            % find the x,y which span the x extend, from the top. this is the
            % "top" of the cup.
            [y,is] = sort(y,'ascend');
            x = x(is);
            xmin = min(x);
            xmax = max(x);
            x_left = find(x==xmin,1,'first');
            x_right = find(x==xmax,1,'first');
            y_tops(iCandidate) = max(y(x_left),y(x_right));
        end
        bboxes(:,[1 3]) = bboxes(:,[1 3])/size(E,2);
        bboxes(:,[2 4]) = bboxes(:,[2 4])/size(E,1);
        bc = boxCenters(bboxes);
        c_score = exp(-(bc(:,1)-.5).^2*10);
        horz_extent = bboxes(:,3)-bboxes(:,1);
        curScore = c_score+horz_extent+ucmStrengths;
        %     showSorted(ims,curScore);
        y_tops = y_tops/size(E,1);
        
        curFeats(1).bbox = bboxes;
        curFeats(1).bc = bc;
        curFeats(1).horz_extent = horz_extent;
        curFeats(1).y_top = y_tops;
        curFeats(1).ucmStrength = ucmStrengths;
        curFeats(1).isConvex = isConvex;
        curFeats(1).salStrength = salStrengths;
        if (debug_)
            %             close all
            [s,is] = sort(ucmStrengths+10*intersectMouth,'descend');
            clf;
            vl_tightsubplot(1,3,1);
            [M,O] = gradientMag(im2single(I),1);
            montage2(cat(3,ims_show{is}));
            vl_tightsubplot(1,3,2); imagesc(rgb2gray(I)+mouth_mask); axis image;
            vl_tightsubplot(1,3,3); imagesc(M); axis image;
            %m = showSorted(ims_show,ucmStrengths);
            pause;
            %                 clf;   subplot(1,2,1); imagesc(I); axis image; colormap('default');
            %                 subplot(1,2,2); imagesc(subUCM); axis image;
        end
        
        curFeats.candidates = candidates;
    end

    function candidates = generate_u_shapes()
        szRange = ([.2 .4 ])*size(I,1);
        alphaRange = [90 112 135 235 248 270];
        % alphaRange = [-90];
        allPts = zeros(4,2,length(szRange)*length(alphaRange));
        k = 0;
        [y,x] = find(mouth_mask);
        bbox = pts2Box([x,y]);
        candidates = {};
        for ix = 1:length(x)
            curPt = [x(ix) y(ix)]';
            %     for iTheta = 1:length(thetaRange)
            curTheta = theta(ix);
            %curTheta = thetaRange(iTheta);
            r = rotationMatrix(pi*curTheta/180);
            u = [cosd(curTheta);sind(curTheta)];
            for iSz = 1:length(szRange)
                curSize = szRange(iSz);
                for iAlpha = 1:length(alphaRange)
                    curAlpha = alphaRange(iAlpha);
                    % construct a candidate u-shape.
                    v1 = r*[-cosd(curAlpha);sind(curAlpha)];
                    v2 = r*[cosd(curAlpha);sind(curAlpha)];
                    pt1 = curPt+u*curSize/2;
                    pt2 = curPt-u*curSize/2;
                    pt1_1 = pt1+2*v1*curSize/2;
                    pt2_1 = pt2+2*v2*curSize/2;
                    k = k+1;
                    allPts(:,:,k) = ([pt2_1,pt2,pt1,pt1_1]');
                    
                    if (all(inImageBounds(size(E),allPts(:,:,k))))
                        candidates{end+1} = fliplr(round((allPts(:,:,k))));
                    end
                    %                 allPts{end+1} = flipud([pt2_1,pt2,pt1,pt1_1]');
                    %
                    if (0 && debug_ &&  (toc > .01))
                        tic
                        clf;
                        imagesc(E);
                        hold on;
                        curPts = allPts(:,:,k);
                        plot(curPt(1),curPt(2),'g*');
                        plot(x(ix),y(ix),'rs');
                        quiver(x(ix),y(ix),5*cosd(theta(ix)),5*sind(theta(ix)),0,'g');
                        drawedgelist({fliplr(curPts)},size(E),2,'r')
                        %                 plot(
                        xlim([1 size(E,2)]);
                        ylim([1 size(E,1)]);
                        
                        
                        %                 pause(.01)
                        drawnow;
                    end
                    %                 tic;
                    %                 end
                end
                %         end
            end
        end
    end
end