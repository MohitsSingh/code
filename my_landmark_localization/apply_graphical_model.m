function [yhat,score,rot] = apply_graphical_model(param,curImg,...
    w_unary,w_prior,w_edges,y)

augmentLoss = nargin == 7;
do_loc_admitted_stuff = true;
nPts = param.nPts;

if (nargin < 4)
    error('need unary,prior,occlusion and binary weight vectors');
end

if (~param.use_pairwise_scores)
    error('parameter not supported');
    w_edges = zeros(size(w_edges));
end


if (~param.use_appearance_feats)
    error('parameter not supported');
    w_unary = zeros(size(w_unary));
end

if (param.infer_visibility)
    location_scores = [location_scores;location_scores];
end
img_orig = curImg;

yHats = {};
rots = param.rotations;
rot_scores = zeros(size(rots));

isFullModel = length(w_unary)>nPts;
if (isFullModel)
    w_detect = reshape(w_unary,[],nPts);
end

if ~param.use_location_prior
    param.location_priors = zeros(size(param.location_priors));
end

for iRot = 1:length(rots)
    pairwise_scores = param.pairwise_scores;    
    curImg = imrotate(img_orig,rots(iRot),'bilinear','crop');
    curImg = imResample(curImg,[param.imgSize param.imgSize],'bilinear');
    if (isFullModel)
        [detection_res] = img_detect2(param,w_detect',curImg);
    else
        [detection_res] = img_detect(param,curImg);
    end
    
    %     end
    box_centers = boxCenters(detection_res(:,1:4));
    box_centers_n = box_centers/size(curImg,1);
    unary_scores = detection_res(:,5:end);
    
    
    
    if 0&&(strcmp(param.phase,'test'))
        figure(1); clf;
        uu = 1;
        %z = visualizeTerm(unary_scores(:,1),box_centers,size(curImg));
        z = visualizeTerm(param.location_priors(:,1),box_centers,size(curImg));
        imagesc2(sc(cat(3,z,0*im2double(curImg)),'prob'));
        %     pause;
        
    end
    if (param.infer_visibility)
        unary_scores = [unary_scores;ones(size(unary_scores))/size(unary_scores,1)];
    end
    

    if (isscalar(w_unary))
        unary_scores = unary_scores*w_unary;
        if param.use_location_prior
            unary_scores = unary_scores+param.location_priors*w_prior;
        end
    elseif (length(w_unary) == size(unary_scores,2)) % partial
        for t = 1:length(w_unary)
            unary_scores(:,t) = unary_scores(:,t)*w_unary(t);
        end
    end % otherwise, full.
    
    if (augmentLoss)
        loss = zeros(size(unary_scores));
        if (param.lossType==2)
            for t = 1:nPts
                loss(:,t) = sum(bsxfun(@minus,box_centers_n,y(t,1:2)).^2,2);
            end
        else
            for t = 1:nPts
                loss(:,t) = sum(abs(bsxfun(@minus,box_centers_n,y(t,1:2))),2);
            end
        end
        unary_scores = unary_scores+loss;
    end
    if (isscalar(w_unary))
        pairwise_scores = pairwise_scores*w_edges;
    else
        for t = 1:size(pairwise_scores,3)
            pairwise_scores(:,:,t) = pairwise_scores(:,:,t)*w_edges(t);
        end
    end
    unary_scores = exp(unary_scores);
    if (any(isinf(unary_scores(:))))
        isinfunary = true;
    end
    
    pairwise_scores = exp(pairwise_scores);
    
    if (any(isinf(pairwise_scores(:))))
        isinfpairwise_scores = true;
    end
    if (param.infer_visibility)
        pairwise_scores = repmat(pairwise_scores,[2 2 1]);
    end
    
    if (do_loc_admitted_stuff)
        if (param.keepDeadStates) % no need to do this...
            unary_scores = unary_scores.*double(param.loc_admitted);
        else
            unary_scores1 = zeros(param.nLocsAdmitted,nPts);
            for t = 1:nPts
                unary_scores1(:,t) = unary_scores(param.loc_admitted(:,t),t);
            end
            unary_scores = unary_scores1;
        end
    end
    
    % best_configuration = decode_structure(param.adj,unary_scores,pairwise_scores);
    adj = full(param.adj)>0;
    adj = adj + adj';
    nStates = size(unary_scores,1);
    [edgeStruct] = UGM_makeEdgeStruct(adj,nStates);
    nodePot = unary_scores';
    
    
    if (param.isTreeStructure)
        [best_configuration] = UGM_Decode_Tree(nodePot,pairwise_scores,edgeStruct);
    else
        [best_configuration] = UGM_Decode_LBP(nodePot,pairwise_scores,edgeStruct);
    end
    if (do_loc_admitted_stuff && ~param.keepDeadStates)
        yhat = zeros(nPts,2);
        for t = 1:nPts
            yhat(t,:) = param.yhats(t,best_configuration(t),:);
        end
        for t = 1:length(best_configuration)
            loc_sel = param.loc_admitted(:,t);
            cur_det_scores = detection_res(loc_sel,t+4);
            rot_scores(iRot) = rot_scores(iRot)+...
                cur_det_scores(best_configuration(t));
            %detection_res(best_configuration(t),t+4);
        end
    else
        yhat = box_centers_n(best_configuration,:);
        for t = 1:length(best_configuration)
            rot_scores(iRot) = rot_scores(iRot)+detection_res(best_configuration(t),t+4);
        end
    end
    yHats{iRot} = yhat;
    
end

% im(param.edges(1,1))
% im(param.edges(1,2))
% imagesc(pairwise_scores(

% find the active edges in this configuration


[score,iscore] = max(rot_scores);
rot = rots(iscore);
yhat = yHats{iscore};
% % %% do some debugging visualization
% if param.debug
%
%     % % %
%     figure(1); clf;
%     subplot(1,2,1);
%     uu = 18;
%     z = visualizeTerm(dd(:,uu),box_centers,size(curImg));
%     imagesc2(sc(cat(3,z,im2double(curImg)),'prob'));
%
%     title('scoreFun');
%     subplot(1,2,2); imagesc2(z);
%     drawnow
%     z = 0;
% end
% %

