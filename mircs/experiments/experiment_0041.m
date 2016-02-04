%% experiment 0041
%% predict location of head from upper body detection.
%% note: this enable you to predict upper body locations in a very good manner.
default_init;
specific_classes_init;
addpath(genpath('/home/amirro/code/3rdparty/cvpr14_saliency_code'));
res = struct('sal',{},'sal_bd',{},'bbox',{},'resizeRatio',{});
opts.show = false;
maxImageSize = 300;
opts.maxImageSize = maxImageSize;
spSize = 40;
opts.pixNumInSP = spSize;
conf.get_full_image = true;
data = struct('faceBox',{},'upperBodies',{},'isvalid',{},'subs',{});
validIndices = 1:length(newImageData);
for ik =1:length(newImageData)
    ik
    k = validIndices(ik);
    curImageData = newImageData(k);
    [I,I_rect] = getImage(conf,curImageData);
    c_ = 'm-';
    if (curImageData.faceScore < -.6)
        if (isempty(curImageData.gt_face));
            faceBox  = [1 1 2 2];
            data(ik).isvalid = false;
        else
            faceBox = curImageData.gt_face.bbox;
        end
        
    else
        faceBox = curImageData.faceBox + I_rect([1 2 1 2]);
        c_ = 'g-';
    end
    ub = curImageData.upperBodyDets;
    
    data(ik).imageIndex = k;
    data(ik).imageID = curImageData.imageID;
    data(ik).faceBox = faceBox;
    data(ik).sz = size2(I);
    data(ik).I_rect = I_rect;
    data(ik).subs = {};
    if (isempty(ub))
        data(ik).isvalid = false;
        continue;
    end
    subs = {};
    sal = foregroundSaliency(conf,curImageData,0);
    
    [nn,mm,areas] = BoxSize(ub);
    [a,b,c] = BoxSize(I_rect);
    ints = col(BoxIntersection2(I_rect,ub));
    
    insides = ints./areas;
    top_dists = ub(:,2)-I_rect(2);
    top_dists = exp(-abs(top_dists./a)*10);
    
    sums = zeros(size(ub,1),1);
    for ib = 1:length(sums)
        curBox = round(ub(ib,1:4));
        curBox = clip_to_image(curBox,I);
        sums(ib) = sum(sum(sal(curBox(2):curBox(4),curBox(1):curBox(3))));
        sums(ib) = sums(ib)/areas(ib);
    end
    
    % TODO: learn these scores, with/without the I_rect bounding box(for
    % comparison)
    scores = insides+.5*top_dists+ub(:,end)+sums+.5*sqrt(areas./c);
    [s,is] = max(scores);
    ub = ub(is,:);
    data(ik).upperBodies = ub;
    
    data(ik).isvalid = true;
    % % %     data(ik).subs = multiCrop(conf,I,ub);
    
    %
    %         clf;imagesc2(I);plotBoxes(faceBox,c_,'LineWidth',2);
    %         plotBoxes(ub,'r--','LineWidth',3);
    %         plotBoxes(I_rect,'y-','LineWidth',2);    %
    % %          [insides top_dists ub(:,end) sums sqrt(areas./c)]
    %         drawnow;
    %         pause
    %         continue;
end
%%
% regression features: bounding box of head detection ( + score?)
% regression output: center of bounding box + scale (x,y,s)
xs = {};
vals = {};
% Hs = {};
% trainData = data(isTrain);
for t = 1:length(data)
    if (~data(t).isvalid)
        continue;
    end
    
    ud = data(t).upperBodies;
    
    scores = ud(:,end);
    sizes = ud(:,3)-ud(:,1);
    offsets = ud(:,1:2);
    fb = data(t).faceBox;
    faceCenter = boxCenters(fb);
    faceSize = (fb(3)-fb(1)+fb(4)-fb(2))/2;
    % 1. normalize w.r.t upper body bounding boxes
    % 2. add as training samples.
    faceCenter = bsxfun(@minus,faceCenter,offsets);
    faceCenter = bsxfun(@rdivide,faceCenter,sizes);
    goods = all(faceCenter>0 & faceCenter < 1,2);
    faceCenter = faceCenter(goods,:);
    sizes = sizes(goods,:);
    
    faceSize = faceSize./sizes;
    %     H = cellfun2(@(x) col(fhog(im2single(x))), data(t).subs);
    %     Hs{t} = cat(2,H{goods})';
    ud = ud(goods,:);
    xs{t} = ud; % n x 5
    %vals{t} = repmat([faceCenter faceSize],size(ud,1),1);  % n x 3
    vals{t} = [faceCenter faceSize];
end

%input_train = cat(1,xs{isTrain & [data.isvalid]});
% input_train = double(cat(1,Hs{isTrain & [data.isvalid]}));
output_train = cat(1,vals{isTrain & [data.isvalid]});
gmfit = gmdistribution.fit(output_train,1);


valids = [data.isvalid];
sel_train = (isTrain & valids);


%%
% show expected location of head/face for each image...
for ik = 1400:length(validIndices)
    ik
    k = validIndices(ik);
    if (isTrain(ik))
        continue;
    end
    curImageData = newImageData(k);
    [I,I_rect] = getImage(conf,curImageData);
    c_ = 'r-'; % red means manual
    if (curImageData.faceScore < -.6)
        faceBox = curImageData.gt_face.bbox;
    else
        faceBox = curImageData.faceBox + I_rect([1 2 1 2]);
        c_ = 'g-';% green means automatic
    end
    clf;
    
    
    subplot(1,2,1);
    imagesc2(I);
    
    plotBoxes(faceBox,c_,'LineWidth',2);
    ub = round(data(ik).upperBodies);
    plotBoxes(ub);
    
    [w h a] = BoxSize(round(ub));
    [xx_,yy_] = meshgrid(1:w,1:h);
    
    xx = xx_/w;yy = yy_/h;
    %pred_s = gmfit.mu(3)*(w+h)/2;
    pred_s = repmat(gmfit.mu(3),size(xx));
    pp = gmfit.pdf([xx(:),yy(:),pred_s(:)]);
    pp = reshape(pp,size(xx));
    %     figure,imagesc2((pp));
    
    R = zeros(size2(I));
    
    yy_ = ub(2)+yy_(:);xx_ = ub(1)+xx_(:);
    inBounds = inImageBounds(I,[xx_ yy_]);
    
    R(sub2ind(size(R),yy_(inBounds),xx_(inBounds))) = pp(inBounds);
    
    sal = foregroundSaliency(conf,curImageData,0);
    E = .5;V = 10;
    U = sal+(R.^E)/E;
    %     figure,imagesc(U);
    [u,iu] = max(U(:));
    
    % predict scale/location
    pred_xy = gmfit.mu(1:2).*[w h]+ub(1:2);
    
    % apply head detection to bounding box...
    
    curSub = data(ik).subs{1};
    %     resizeFactor = 128/size(curSub,1);
    %     curSub = imResample(curSub,resizeFactor);
    pred_s = gmfit.mu(3)*(w+h)/2;
    pred_bb = inflatebbox([pred_xy pred_xy],pred_s,'both',true);
    plotBoxes(pred_bb,'g--','LineWidth',2);
    opts.pixNumInSP = 50;
    %     opts.maxImageSize = 200;
    %     [sal,sal_bd,resizeRatio] = extractSaliencyMap(im2uint8(curSub),opts);
    clf; subplot(1,2,1);imagesc2(curSub);    
    L = load(j2m('~/storage/s40_upper_body_faces/',curImageData.imageID));
    subplot(1,2,2);imagesc2(curSub);plotBoxes(L.res(1,:));
    plotBoxes(faceBox(1:4)-ub([1 2 1 2]),c_,'LineWidth',2);
    pause;
    continue;
end

%%
cd /home/amirro/code/3rdparty/voc-release5/

%%
for t =  218:length(data)
    if (~data(t).isvalid)
        continue;
    end
    t
    curSub = data(t).subs{1};
    %         curSub = imResample(curSub,128/size(curSub,1));    
    [~,face_boxes] = face_detection(curSub);    
    clf;imagesc2(curSub); hold on; plotBoxes(face_boxes);    
    %         curSub = curSub(1:end*.75,:,:);    
    %         [dets, boxes] = imgdetect(curSub, model, -1);
    
    pause
    
end


