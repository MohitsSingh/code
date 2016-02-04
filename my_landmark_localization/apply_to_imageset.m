function [gts,preds,stats] = apply_to_imageset(imgs,phis,model,param,kp_sel,learningType,toShow);
stats = [];
param.phase = 'test';
imgSize = param.imgSize;
nImages = length(imgs);
% nPts = param.nPts;
% nEdges = param.nEdges;
gts = cell(nImages,1);
preds = cell(nImages,1);
ticID = ticStatus('predicting on image set...',.5,.5);
% param.pairwise_scores = sparse(param.pairwise_scores);

for iImg = 1:nImages
    %     iImg
    curImg_orig = imgs{iImg};
    
    bestScore = -inf;
    for v = 1
        curImg = curImg_orig;
        r = [1 1 fliplr(size2(curImg))];
        curImg = cropper(curImg,round(inflatebbox(r,v,'both',false)));
        curImg = imResample(curImg,[imgSize imgSize]);
        w_prior = [];
        switch learningType
            case {'nolearn','minimal'}
                w_unary = model.w(1);
                w_prior = model.w(2);
                w_edges = model.w(3);
            case 'partial'
                w_unary = model.w(1:param.nPts);
                w_edges = model.w(param.nPts+1:end,:);
%                 error('partial learning - location prior not implemented')
            case 'full'
%                 error('full learning - location prior not implemented')
                n = (31*(param.windowSize/param.cellSize)^2)*param.nPts;
                w_unary = model.w(1:n);
                w_edges = model.w(n+1:end);
        end
        
        
        
        
        [xy_pred,score,rot] = apply_graphical_model(param,curImg,...
            w_unary,w_prior,w_edges);
        [xy_pred1,score1,rot1] = apply_graphical_model(param,flip_image(curImg),...
            w_unary,w_prior,w_edges);
        
        toFlip = false;
        if (score1 > score)
            curImg = flip_image(curImg);
            toFlip = true;
            xy_pred = xy_pred1;
            score = score1;
            rot = rot1;
        end
        score
        if (bestScore < score)
            bestScore=score;
        else
            continue;
        end
        
        %         assert(all(xy_pred_0(:)==xy_pred(:)));
        %         assert(all(score_0(:)==score(:)));
        %         assert(all(rot_0(:)==rot(:)));
        
        %     end
        if (~isempty(phis))
            xy_gt = squeeze(phis(iImg,kp_sel,1:2))/size(curImg_orig,1);
            gts{iImg} = xy_gt;
        end
        % rotate backwards...
        xy_pred_rot = rotate_pts(xy_pred,-pi*rot/180,[.5 .5]);
        
        
        % % %     figure(2);
        % % %     clf; imagesc2(imrotate(curImg,rot,'bilinear','crop'));
        % % %     plotPolygons(xy_pred*size(curImg,1),'r.');
        % % %     pause;continue
        
        
        preds{iImg} = xy_pred_rot;
        if (~isempty(phis))
            xy_gt1 = xy_gt*size(curImg,1);
        end
        xy_pred_1 = xy_pred_rot*size(curImg,1);
        
        if (toShow)
            figure(2)
            
            clf; imagesc2(curImg);
            colors = 'rgbcmykr';
            %         for u = 1:size(xy_gt,1)
            %             plotPolygons(imgSize* squeeze(param.yhats(u,:,:)),[colors(u) '.'],'MarkerSize',3);
            %         end
            %         clf; imagesc2(curImg+repmat();
            %     %     %     set(hh,'ShowArrowHead',false)
            %     %     plotPolygons(xy_pred1,'r.','LineWidth',2);%
            %     %     param.pwise_factor = .5;
            %     %     [xy_pred] = apply_graphical_model(param,curImg);
            %     %     xy_pred = detect(param,model,curImg);
            
            plotPolygons(xy_pred_1,'cd','LineWidth',2);
            % % %         plotPolygons(xy_gt1,'g.','LineWidth',2);
            % % %         hh = quiver2(xy_gt1,xy_pred_1-xy_gt1,0,'b-','LineWidth',2, 'ShowArrowHead','off');
            gplot2(param.adj,xy_pred_1,'g-','LineWidth',2);
            % % %         showCoords(xy_pred_1);
            
            drawnow
            pause;
        end
        %     w
        tocStatus(ticID,iImg/nImages);
        
        %     % % % %
        %     % % % %     showCoords(xy_gt);
        %     pause;
    end
end
% preds = cat(3,preds{:});
% gts = cat(3,gts{:});

% compute per-image error per keypoint and avg the results.
if (~isempty(phis))
    nKP = length(kp_sel);
    kp_errors = zeros(nKP,nImages);
    for iImg = 1:nImages
        curError = sum( (preds{iImg}-gts{iImg}).^2,2).^.5;
        kp_errors(:,iImg) = curError;
    end
    
    stats.kp_errors = kp_errors;
    stats.mean_error = mean(kp_errors,1);
end

%%
% close all;figure,plotPolygons(xy_pred,'.'); title('orig');xlim([0 1]);ylim([0 1]);
% figure,plotPolygons(xy_pred_rot,'.'); title('rotated');xlim([0 1]);ylim([0 1]);