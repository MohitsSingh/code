function curPred = detect_keypoints(I,p0,p1,featureExtractor,wSize,toJoin,toDebug)
if nargin < 6
    toJoin = false;
end
if nargin < 7
    toDebug=  false;
end
globalFeats = featureExtractor.extractFeaturesMulti(I,false);
X = normalize_vec(globalFeats);
cur_preds0 = squeeze(apply_predictors(p0,X,1))';

% cur_preds0 = cur_preds0;%+rand(size(cur_preds0))*15;

prevPred = cur_preds0;
if toDebug
    clf; imagesc2(I); plotPolygons(cur_preds0,'ms','LineWidth',5);
end
if isempty(p1)
    return
end

nSteps = 10;
stepSize = (1./(1:nSteps)).^.5;
delta_norms = zeros(nSteps,1);
for u = 1:nSteps
    curBox = round(inflatebbox([prevPred prevPred],wSize,'both',true));
    patch_local = cropper(I,curBox);
    if toJoin
        feats_local = [globalFeats;...
            featureExtractor.extractFeaturesMulti(patch_local,false)];
    else
        feats_local = featureExtractor.extractFeaturesMulti(patch_local,false);
    end
    feats_local = normalize_vec(feats_local,1);
    pred_delta =  squeeze(apply_predictors(p1,feats_local,1))';
    
    %pred_delta = pred_delta*stepSize(u);
    curPred = prevPred+pred_delta*stepSize(u);
    delta_norms(u) = norm(pred_delta);
    
    %     imagesc2(I);
    if toDebug
        plotPolygons(prevPred,'ms');
        plotPolygons(curPred,'gs');
        %     plotBoxes(curBox)
        quiver(prevPred(1),prevPred(2),pred_delta(1),pred_delta(2));
        if (mod(u,3)==0)
            %         dpc(.1)
            drawnow
        end
    end
    prevPred = curPred;
    if u >=5
        diff_norm =mean(delta_norms(u-4:u));
        if toDebug
            disp(diff_norm)
        end
        if diff_norm < 1
            %                 break
        end
    end
end

