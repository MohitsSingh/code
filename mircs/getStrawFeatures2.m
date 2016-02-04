function [frs,pss,Zs,strawBoxes,xys,thetas] = getStrawFeatures2(conf,imageDataStruct,imageSubset,debug_,ws,b,fhog1,props)
if (nargin < 4)
    debug_ = false;
end
conf.get_full_image = true;
cur_t = imageDataStruct.labels;
if (nargin < 3 || isempty(imageSubset))
    imageSubset = 1:length(cur_t);
end

dTheta = 10;
thetaRange = 0:dTheta:180-dTheta;

fb2 = makeFilterBank(dTheta,thetaRange);

frs = {};
pss = {};
Zs = {};
bbs = {};
xys = {};
thetas = {};
if (nargin < 7)
    fhog1 = @(x) fhog(im2single(x),8,9,.2,0);
end

close all;

doFollowing = false;

for q = 1:length(imageSubset)
    %     if (~cur_t(k)) continue; end
    k = imageSubset(q)
%     k = 505
%     if (k ~= 489)
%         continue;
%     end
    imageInd = k;
    currentID = imageDataStruct.imageIDs{imageInd};
    I = getImage(conf,currentID);
    faceBoxShifted = imageDataStruct.faceBoxes(imageInd,:);
    lipRectShifted = imageDataStruct.lipBoxes(imageInd,:);
    
    if (debug_)
        clf;
        subplot(2,3,1);
        imagesc(I); axis image; hold on;
        plotBoxes2(faceBoxShifted([2 1 4 3]));
        plotBoxes2(lipRectShifted([2 1 4 3]),'m');
    end
                    
    box_c = round(boxCenters(lipRectShifted));
    
    % get the radius using the face box.
    [r c] = BoxSize(faceBoxShifted);
    boxRad = (r+c)/2;
    
  bbox = lipRectShifted;
  bbox = makeSquare(bbox);
  bbox([2 4]) = bbox([2 4]) + (bbox(4)-bbox(2))/2;
    bbox = round(bbox);
    
    if (debug_)
        plotBoxes2(bbox([2 1 4 3]),'g');
    end
    
    if (any(~inImageBounds(size(I),box2Pts(bbox))))
        continue;
    end
    I_sub = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
    I_sub = imresize(I_sub,[50 NaN],'bilinear');
    I_sub = rgb2gray(I_sub);
    
    % apply the filter-bank and look for strong double responses...
    fr = abs(FbApply2d(I_sub,fb2,'same'));
    % clip borders.
    ddd = 3;
    fr(1:ddd,:,:) = 0;
    fr(end-ddd+1:end,:,:) = 0;
    fr(:,1:ddd,:) = 0;
    fr(:,end-ddd+1:end,:) = 0;
    if (debug_)
        subplot(2,3,2);
        imagesc(I_sub); axis image; hold on; colormap(gray);
    end
    
    % crop responses to ranges of angles.
    sz = size(I_sub);
    cx = sz(2)/2;
    cy = 1;
    [xx,yy] = meshgrid(1:sz(2),1:sz(1));
    theta = 180-180*atan2(yy-cy,xx-cx)/pi;
    maxRad = 15;
    radCap = ((yy-cy).^2+(xx-cx).^2).^.5 <= maxRad;
    fr2 = fr;
    
%     frs = filterMultiscale(I,fb)
    
    wTheta = dTheta/2;
    %     z = zeros(length(thetaRange),1);
    doCap = true;
    II = imresize(rgb2gray(I(bbox(2):bbox(4),bbox(1):bbox(3),:)),...
        [100 100],'bicubic');
    ps = phasesym(II);
    ps = imresize(ps,size(I_sub));
    if (doCap)
        minTheta = 30;
        for qq = 1:length(thetaRange)
            if (thetaRange(qq) < (180-minTheta) && thetaRange(qq) > minTheta)
                s = (theta <= thetaRange(qq)+wTheta) & (theta>=thetaRange(qq)-wTheta);
                s = imdilate(s,ones(1,9));
                fr2(:,:,qq) = fr(:,:,qq).*s.*radCap;
            else
                fr2(:,:,qq) = 0;
            end
        end
    end
    
    frs{k} = single(fr2);
    pss{k} = single(ps);
    
    c_ = bsxfun(@times,fr2,ps);
    c_sum = (squeeze(sum(sum(c_,1),2))).^2;
    c_sum = c_sum/sum(c_sum);
    
    [m,im] = max(c_(:));
    
    [ii,jj,kk] = ind2sub(size(fr2),im);
    cd_ = maxRad*cosd(180-thetaRange(kk));
    sd_ = maxRad*sind(180-thetaRange(kk));
    R = rotationMatrix(pi*(180-thetaRange(kk))/180);
    mx = 1; my = .5;
    boxSide = 15;
    x = mx*[-1 2 2 -1]*boxSide; y =  my*[-1 -1 1 1]*boxSide;
    xy = R*[x;y];
    xy = bsxfun(@plus,xy,[jj;ii]);
    thetas{k} = 180-thetaRange(kk);
    % map it back to the original image.
    bsize = segs2vecs(bbox); %(width,height)
    resizeFactor = bsize(2)/size(I_sub,1);
    xy = xy*resizeFactor;
    xy = bsxfun(@plus,xy,bbox(1:2)');
    xys{k} = xy;
    %remap this area to a canonical frame.
    stepSize = 1/40;    
    [X,Y] = meshgrid(0:stepSize:mx,0:stepSize:my);
    x_t = mx*[0 1 1 0]';
    y_t = my*[0 0 1 1]';
    T = cp2tform([x_t y_t],xy','affine');
    [X_,Y_]= tformfwd(T,X,Y);
    Z = cat(3,interp2(I(:,:,1),X_ ,Y_,'bilinear'),...
        interp2(I(:,:,2),X_ ,Y_,'bilinear'),...
        interp2(I(:,:,3),X_ ,Y_,'bilinear'));
    Z(isnan(Z(:))) = 0;
    Zs{k} = Z;
    p = pts2Box([X_(:) Y_(:)]);
    strawBoxes{k} = p;
    if (doFollowing)
        subplot(2,3,1);
        bb = getCandidateBoxes(p,props{k});
        if (debug_)
            hold on; plotBoxes2(bb(:,[2 1 4 3]),'y');
        end
    end
    
    if (debug_)
        subplot(2,3,2);
        title(num2str(m));
        hold on;
        quiver(jj,ii,...
            cd_,sd_,0,'g','LineWidth',3);
        hold on; plot(jj,ii,'r+');
        subplot(2,3,3);imagesc(Z); axis image;
        
        p = inflatebbox(p,2,'both');
        subplot(2,3,1); %imagesc(I); axis(p([1 3 2 4]));
        hold on; plot(xy(1,[1:4 1]),xy(2,[1:4 1]),'r','LineWidth',2);
        Z = im2single(Z);
        
        subplot(2,3,3); hold on;
        ff = fhog1(Z);
        subplot(2,3,6); imagesc(hogDraw(ff.^2,15,1));axis image;
        
        subplot(2,3,1); hold on; 
        
        E = edge(im2double(rgb2gray(I)),'canny');
        subplot(2,3,4); imagesc(E); axis image;
         xy_c = mean(xy'); % clear all edges which are too far from the center point.
         [edge_y,edge_x] = find(E);
         f = l2([edge_x edge_y],xy_c) > (150^2);
         E(sub2ind(size(E),edge_y(f), edge_x(f))) = 0;
        [edgelist edgeim] = edgelink(E, []);
        seglist = lineseg(edgelist,3);
        seglist = segs2seglist(seglist2segs(seglist)); % break into single component segments.
        drawedgelist(seglist,size(E),2,'rand');
        % = edgelist2image(seglist,size(E));
        segs = seglist2segs(seglist);
  
%         figure,imagesc(E); axis image;
%         drawedgelist(seglist1,size(E),2,'rand');
        [EE,allPts] = paintLines(zeros(size(E)),segs);
        inds = makeInds(allPts);
       
        allPts = cat(1,allPts{:});
        dists = l2(xy_c,allPts).^.5;
%         figure,imshow(EE);
        subplot(2,3,1);
        hold on;
        plot(xy_c(1),xy_c(2),'m*');
%         plot(allPts(dists<10,1),allPts(dists<10,2),'r+');
        [m,im] = min(dists); 
                
        segs_c = unique(inds(dists<5));        
        % remove the edges that disagree with the direction found by the
        % gabor filter. 
        vecs_ = segs2vecs(segs(segs_c,:));
        vec_dirs = normalize_vec(vecs_')';
        prods = vec_dirs* [cosd(thetas{k});sind(thetas{k})]; % want the prod to be large
        segs_c = segs_c(abs(prods) > .7);
%         segs_c = inds(im); % find the nearest edge.
         subplot(2,3,5);
%         figure,
        imshow(EE);
        hold on;
        drawedgelist(seglist(segs_c),size(E),2,'r');
        %curSeg = fixSegs(segs(segs_c,:));
        [y,iy] = max(segs(segs_c,[1 3]),[],2);
        x = segs(segs_c,[iy*2]);
        
        plot(x,y,'rs','MarkerSize',6,'MarkerFaceColor','red');
        % find the edge which agrees with the orienation.
        
        
%         clf,imagesc(EE>0);
%         hold on;
%         drawedgelist(seglist(inds(dists<10)),size(E),2,'rand');
%         
        
        %[D,IDX] = bwdist(EE);
        %         curScore = dot(ws,ff(:))-b;
        %         title(num2str(curScore));
        %         subplot(2,3,5); montage2(c_);
        pause;
    end
end

end
function fb2 = makeFilterBank(dTheta,thetaRange)
fb = FbMake(2,4,1);
fb = squeeze(fb(:,:,3));
b = zeros(dsize(fb,1:2));
doGabor1 = true;
if (doGabor1)
    b(1:end,ceil(size(b,2)/2)) = 1;
    fb2 = zeros([size(fb) length(thetaRange)]);
    a = zeros(7);
    a(3,4) = 1;
    a(5,4) = -1;
    for k = 1:length(thetaRange)
        q = imrotate(b',thetaRange(k),'bicubic','crop');
        fb2(:,:,k) = FbApply2d(q,imrotate(fb(:,:,1),(k-1)*dTheta,'bicubic','crop'),'same',0);
        fb2(:,:,k) = fb2(:,:,k)/sum(sum(abs(fb2(:,:,k))));
        %             fb3(:,:,k) = imrotate(a,(k-1)*dTheta,'bicubic','crop');
    end
end
end


