
d = defaultPipelineParams();
%d.roiParams.absScale=-1;
m = 3;
n = m*m;
u = vl_colsubset(1:length(fra_db),n,'random');
% u =  [21    31    35   101   112   113   144   268   352];
clf;
%roiParams.infScale = 2;
roiParams.infScale = 2;
roiParams.absScale = 200;
roiParams.centerOnMouth = 1;
% roiParams = rmfield(roiParams,'roiBox');
% roiParams = rmfield(roiParams,'faceBox');
debugging = true;
problem_types = makeStringEnum({'no roi mask','mask out of mouthbox','no mouth center'});
if ~debugging
    seg_bench = struct('valid',{},'segResults',{},'problem_type',{});
end
conf.get_full_image = true;

shrinks = [1 .7 .5 .25 ];
for k = 1:10:length(fra_db)
    % for ik = 1:length(fra_db)
    %     k = ii(ik);
    k
    imgData = fra_db(k);
    seg_bench(k).valid = false;
    if strcmp(imgData.class,'phoning'),
        continue
    end
    if (~imgData.isTrain)
        continue
    end
    I = getImage(conf,imgData);
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    hasGroundTruth = false;
    % 2. remove "trivial" regions
    if ~isfield(gt_graph{2},'roiMask')
        seg_bench(k).problem_type = problem_types.no_roi_mask;
        continue
    end
    m = gt_graph{2}.roiMask;
    if isempty(m) || nnz(m)==0
        seg_bench(k).problem_type = problem_types.no_roi_mask;
        continue
    end
    
    [I_sub,faceBox,mouthBox] = getSubImage2(conf,imgData);
    gtMask = cropper(m,mouthBox);
    if (nnz(gtMask)==0)
        seg_bench(k).problem_type = problem_types.mask_out_of_mouthbox;
        continue
    end
    
    [kp_preds,goods] = loadKeypointsGroundTruth(imgData,d.requiredKeypoints);
    confidences = kp_preds(:,3);
    
    if (~goods(3))
        seg_bench(k).problem_type = problem_types.no_mouth_center;
        continue;
    end
    
    seg_bench(k).valid = true;
    
    kp_preds = kp_preds(goods,1:2);
    %     I = getImage(conf,imgData);
    %     clf;
    %     subplot(1,2,1); imagesc2(I); plotBoxes(mouthBox);
    %     plotPolygons(kp_preds,'b.','LineWidth',2,'MarkerSize',10)
    kp_preds = kp_preds-repmat(mouthBox(1:2),size(kp_preds,1),1);
    
    %     subplot(1,2,2);
    %     clf;imagesc2(I_sub);hold on;
    %     plotPolygons(kp_preds,'b.','LineWidth',2,'MarkerSize',10)
    [candidates,ucm2,isvalid] = getCandidateRegions(conf,imgData,I_sub);
    
    % now start finding the best overlap given the extent
    s = size2(I_sub);
    xy = [kp_preds(3,:) kp_preds(3,:)]
    segResults = struct('shrink',{},'bestOvp',{},'candidatesLeft',{},'gtLeft',{});
    for iShrink = 1:length(shrinks)
        curShrink = shrinks(iShrink)
        curRect = inflatebbox(xy,s*curShrink,'both',true);
        curRect = round(clip_to_image(curRect,I_sub));
        masks = multiCrop2(candidates.masks,curRect-1);
        masks = removeDuplicateRegions(masks);
        curGT = cropper(gtMask,curRect);
        [ovps ints uns] = regionsOverlap3(masks,{curGT});
        segResults(iShrink).shrink = curShrink;
        segResults(iShrink).bestOvp = max(ovps);
        segResults(iShrink).candidatesLeft = length(masks)/length(candidates.masks);
        segResults(iShrink).gtLeft = nnz(curGT)/nnz(gtMask);
        %         mm = 1;
        %         nn = 3;
        % %         figure(1);
        %         clf;
        %         subplot(mm,nn,1);
        %         displayRegions(I_sub,gtMask,[],'dontPause',true);
        %         title('original +gt');
        %         I_subsub = cropper(I_sub,curRect);
        % %         figure(2); clf;
        %         subplot(mm,nn,2);
        %         displayRegions(I_subsub,curGT,[],'dontPause',true);
        %         title('cropped + gt');
        % %         figure(3); clf;
        %         subplot(mm,nn,3);
        %         displayRegions(curGT,masks,ovps,'maxRegions',1);%
    end
    seg_bench(k).segResults = segResults;
end
%% plot the segmentation performance vs the extent left around the mouth area
valids = [seg_bench.valid];
s1 = {};
inds = {};
for t=1:length(seg_bench)
    s = seg_bench(t);
    if ~isempty(s.valid) && s.valid
        s1{end+1} = s;
        inds{end+1} = t;
    end
end
s1 = [s1{:}];
S = cat(1,s1.segResults);
ovps = reshape([S.bestOvp],size(S));
pLeft = reshape([S.candidatesLeft],size(S));
gtLeft = reshape([S.gtLeft],size(S));


%%
% hist(ovps(:,3)-ovps(:,1),20)
figure(1);
[n,x] = hist(bsxfun(@minus,ovps(:,2:end),ovps(:,1)),10);
n = bsxfun(@rdivide,n,sum(n));
bar(x,n);
% plot(x,cumsum(n));
legend({'.75','.5.','.25'});
xlabel('overlap improvement');
ylabel('%  images');
title('object overlap improvement vs. allowed extent around mouth');

%mean(gtLeft)
[deltas,ideltas] = sort(ovps(:,end)-ovps(:,1));
ii = [inds{ideltas}];

%%
figure(2);
[n,x] = hist(gtLeft(:,2:end),10);
n = bsxfun(@rdivide,n,sum(n));
bar(x,n);
% plot(x,cumsum(n));
legend({'.75','.5.','.25'});
xlabel('gt left');
ylabel('%  images');

% figure(3),plot(gtLeft(:),ovps(:),'r.');
% explore the cases where the improvement was bad.
figure(3)
clf;
[n,x] = hist(ovps);
plot(x,bsxfun(@rdivide,cumsum(n),sum(n)));
legend({'1','.75','.5.','.25'});
