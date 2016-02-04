classdef MUB_CfgSM
% Three types of models at three levels:
%   smModel: single configuration, multiple locations and scales
%   cmModel: multiple smModels
%   mmModel: multiple cmModels
% This class is for smModel
% smModel: configuration model with the following fields
%   ubUnions(j).relBox: 4 numbers for relative location of ubUnion wrt to im43box
%   ubUnions(j).w: 3 numbers for weights wx, wy, wscale
%   parts(i).relBox: 4 numbers for relative location of an ub wrt to ubUnion
%   parts(i).w: 4 numbers for weights wx, wy, wscale, wscore
%   Make sure wx, wy of ubUnions(j) and parts(i) are positive.
%       If they are 0, dt_ramanan will crash. If they are < 0, the function returns junk
% By: Minh Hoai Nguyen (minhhoai@robots.ox.ac.uk)
% Created: 06-Apr-2013
% Last modified: 20-Mar-2014

    %% Important methods for running the code
    methods (Static)        
        % cfgBoxes: k*(1+5*nUb+5) matrix, each row is a detected configuration
        %   cfgBoxes(i,:) is: total-score, 
        %                     nUb*[x1, y1, w, h, ubDetScore], (for ubs), 
        %                     [x1, y1, w, h, ubUnionRefId] (for ubUnion)
        function cfgBoxes = predict(smModel, unaryScore)
            imH = unaryScore.imH;
            imW = unaryScore.imW;
            ubScores = unaryScore.scores;
            scales   = unaryScore.scales;
            rootSize = unaryScore.rsize;
            xyOffsets = unaryScore.xyOffsets;
            
            im43rect = MUB_CfgBox.get43box(imH, imW); % x1, y1, x2, y2
            im43box = [im43rect(1:2), im43rect(3:4) - im43rect(1:2) + 1]; % x1, y1, w, h            
            im43boxH = im43box(4);
            im43boxW = im43box(3);
            
            MIN_SCALERATIO = 0.8; % scale-ratio of a part and reference part cannot be smaller
            
            ubUnions = smModel.ubUnions; 
            parts   = smModel.parts;
            nUbUnion = length(ubUnions);
            nPart = length(parts);
                
            % work out the muliplication from the relative heights
            relBoxes = cat(1, parts(:).relBox);
            relHeights = relBoxes(:,4);
            
            % NOTE: the smModel MUST have same aspect ratio for all ubUnions!!!
            % i.e., ubUnions(i).relBox(3)/ubUnions(i).relBox(4) is a constant
            ubUnionHeightMul = 1/max(relHeights)*rootSize(1);            
            ubUnionAspectRatio = 4/3*ubUnions(1).relBox(3)/ubUnions(1).relBox(4); 
            ubUnionWidthMul = ubUnionHeightMul*ubUnionAspectRatio;
            
            % work out the center and heigh the refUbUnion
            relUbUnions = cat(1, ubUnions(:).relBox);
            refUbUnions = ML_RectUtils.relBoxes2absBoxes(im43box, relUbUnions');
            refUbUnionCenters = refUbUnions(1:2,:) + (refUbUnions(3:4,:)-1)/2;
            refUbUnionHeights = refUbUnions(4,:);
            minRefUbUnionHeight = min(refUbUnionHeights);
            maxRefUbUnionHeight = max(refUbUnionHeights);
                                    
            nScale = length(scales);   
            cfgBoxes = cell(1, nScale); % output
            
            % a scale level is valid if it contains at least one detection score > thresh
            % This function assuems transformed scores through a logistic function
            % A score of -1 before transformation corresponds to 0.4502 after transformation
            % For cascade detection, a score of -1 is deemed as -inf, and the transformed -inf is 0.
            % In both case, use thresh = 0.4502 is a very safe value.
            % A score of -1 before transformation is already low to not miss any good detection
            % Valid or invalid is only applicable for ubs, not ubUnion
            invalidLevel = false(1, nScale);
            validThresh = 0.4502;
            for i=1:nScale
                invalidLevel(i) = isempty(find(ubScores{i} > validThresh,1));
            end;            
            
            % To speed things up, one can skip scales
            for i=1:nScale % for each scale of the ubUnion, each scale can be processed independently                            
                unionScale = scales(i);
                xyOffset = xyOffsets{i};
                unionUbH = ubUnionHeightMul*unionScale; % height and width of the ubUnion, in image scale
                unionUbW = ubUnionWidthMul*unionScale;
                
                if unionUbH > (1/MIN_SCALERATIO)*maxRefUbUnionHeight || ...
                   unionUbH < MIN_SCALERATIO*minRefUbUnionHeight
                    continue;
                end;
                                
                [nFR, nFC] = size(ubScores{i}); % number of locations                
                nFRFC = nFR*nFC;
                frfcVec = (1:nFRFC)';
                % consider the ubUnion to have the same centers as the boxes coresponds to
                % ubScores{i}. See partRect function

                cfgScore = 0;
                % dynamic programming info
                dpInfo = struct('Ix', [], 'Iy', [], 'Is', [], 'ubScore', cell(1, nPart));
                for j=1:nPart % for each ub part
                    part_j = parts(j);
                    part_j_relBox = part_j.relBox;
                    part_j_w = part_j.w;
                    [score_tmp2, Ix_tmp2, Iy_tmp2, ubScore_tmp2] = deal(zeros(nFR, nFC, i));                    
                    
                    % center and scale of the reference default part
                    % center is specified wrt to the center of ubUnion
                    % For simplied expression, the computation of refCenter is no longer needed
                    % but it should be left here for reference: 
                    % refCenter = [parts(j).relBox(1)*unionUbW, parts(j).relBox(2)*unionUbH];
                    refHeight = part_j_relBox(4)*unionUbH;
                    refScale  = refHeight/rootSize(1);                                        
                    
                    wx = part_j_w(1);
                    ax = wx/(ubUnionWidthMul^2);
                    bx = wx*2*(part_j_relBox(1))/ubUnionWidthMul;
                    biasx = wx*part_j_relBox(1)^2;
                    
                    wy = part_j_w(2);
                    ay = wy/(ubUnionHeightMul^2);
                    by = wy*2*(part_j_relBox(2))/ubUnionHeightMul;
                    biasy = wy*part_j_relBox(2)^2;
                    
                    biasxy = biasx + biasy;                    
                    ws = part_j_w(3);
                    wscore = part_j_w(4);
                    
                    for u=1:i % consider ubs detected at finer scale                         
                        ubScale = scales(u);                        
                        if invalidLevel(u) || ...
                           ubScale > (1/MIN_SCALERATIO)*refScale || ...
                           ubScale < MIN_SCALERATIO*refScale 
                            score_tmp2(:,:,u) = -inf; 
                            continue;
                        end;
                                                
                        % Derivation for the distance transform weights
                        % refCenter(1) is the distance in the image scale
                        % The actual distance in dt space is refCenter(1)/unionScale. So we want:
                        % ax*(x-u + refCenter(1)/unionScale)^2 = 
                        % wx/unionUbW^2*(x_in_im - u_in_im + refCenter(1)) % normalized by unionUbW;
                        % with wx is the weight for x component
                        % x, u: in the coordiate in the distance transform space
                        % x_in_im, u_in_im: in the coordinate of original image
                        % Actually: x_in_im - u_in_im = (x-u)*unionScale
                        % Thus ax = wx*(unionScale/unionUbW)^2;
                        % bx = 2*ax*refCenter(1)/unionScale
                        % with wx = parts(j).w(1);
                        % We work out the math simply the expressions:                        
                        % wx = part_j_w(1);                        
                        % ax = wx/(ubUnionWidthMul^2);
                        % bx = wx*2*(part_j_relBox(1))/ubUnionWidthMul;
                        % biasx = wx*part_j_relBox(1)^2;
                        % 
                        % wy = part_j_w(2);
                        % ay = wy/(ubUnionHeightMul^2);
                        % by = wy*2*(part_j_relBox(2))/ubUnionHeightMul;
                        % biasy = wy*part_j_relBox(2)^2;
                        % It turns out that these weight do not depend on the scale of u
                        % so I move them outside the loop
                        
                        % downsample the ubScores{u}                        
                        ubScores_u = MUB_CfgBox.resizeDenseScore(ubScores{u}, ubScale, xyOffsets{u}, ...
                                            nFR, nFC, unionScale, xyOffset, rootSize); 
                        
                        % distance transform, max over spatial deformation for each scale
                        % score_tmp(x,y) = max_{u,v}
                        %   score(u,v) - (ax*(x-u)^2 +bx*(x-u) + ay*(y-v)^2 + by*(y-v))
                        % Do not use dt function from voc-release.                        
                        [score_tmp,Ix_tmp2(:,:,u),Iy_tmp2(:,:,u)] = dt_ramanan(wscore*ubScores_u, ax, bx, ay, by);
                        ubScore_tmp2(:,:,u) = ubScores_u;                        
                        
                        % adjust for scale deformation
                        % scales are incremental linearly in log-scale
                        % coudl alos use -ws*(ubScle - refScale)^2/unionH^2;                        
                        score_tmp2(:,:,u) = score_tmp - (biasxy + ws*(log(ubScale/refScale))^2);
                    end;       
                    [maxScore, maxIdx] = max(score_tmp2, [], 3); % max over the scale                    
                    cfgScore = cfgScore + maxScore;                    
                                                            
                    % i2 = reshape(1:nFRFC, nFR, nFC) + nFRFC*(maxIdx -1);                    
                    maxIdx = maxIdx(:); i2 = frfcVec + nFRFC*(maxIdx - 1);                    
                    
                    dpInfo(j).Ix = Ix_tmp2(i2); % best x,y that correspond to the best scale
                    dpInfo(j).Iy = Iy_tmp2(i2);
                    dpInfo(j).Is = scales(maxIdx); % scale that gives maximum score at each location
                    try
                        %i3 = sub2ind([nFR, nFC, i], dpInfo(j).Iy, dpInfo(j).Ix, maxIdx); 
                        i3 = nFRFC*(maxIdx - 1) + nFR*(dpInfo(j).Ix - 1) + dpInfo(j).Iy; % faster than the above
                        dpInfo(j).ubScore = ubScore_tmp2(i3); % unary score from ub detection                    
                    catch %#ok<CTCH>
                    end;
                end;  
                                
                % Work out the center of ubUnion in the image space
                [X, Y] = meshgrid(1:nFC, 1:nFR);
                X = X(:);
                Y = Y(:);
                cfgScore = cfgScore(:); 
                
                % center of the ubUnion
                [CX, CY] = MUB_CfgBox.partCenter(X, Y, unionScale, xyOffset, rootSize);
                                
                % For each x,y location, find the best reference ubUnion (lowest deformation).
                % The number of nUbUnion is small, don't use repmat, which will be slower                
                ubUnionDeforms = zeros(nFC*nFR, nUbUnion);
                for v=1:nUbUnion
                    ubUnion_v = ubUnions(v);
                    wx = ubUnion_v.w(1);
                    wy = ubUnion_v.w(2);
                    ws = ubUnion_v.w(3);
                    ubUnionDeforms(:,v) = wx/(im43boxW^2)*(CX - refUbUnionCenters(1, v)).^2 ...
                        + wy/(im43boxH^2)*(CY - refUbUnionCenters(2, v)).^2 ...
                        + ws*(log(unionUbH/refUbUnionHeights(v)))^2;
                end;
                [ubUnionDeform, ubUnionRefIds] = min(ubUnionDeforms, [], 2);                
                cfgScore = cfgScore - ubUnionDeform;
                         
                if strcmpi(smModel.thresh, 'maxOnly')
                    [maxScore,I2] = max(cfgScore);                    
                    if maxScore == -inf;
                        continue;
                    end;
                else
                    I2 = find(cfgScore > smModel.thresh);
                end
                
                if isempty(I2)
                    continue;
                end;
                
                CX2 = CX(I2); CY2 = CY(I2);
                ubUnionBoxes = [CX2 - (unionUbW-1)/2, CY2 - (unionUbH-1)/2];
                ubUnionBoxes(:,3) = unionUbW;
                ubUnionBoxes(:,4) = unionUbH;
                ubUnionRefIds = ubUnionRefIds(I2);
                                
                cfgScore2 = cfgScore(I2); 
                
                % backtrack to find the ubs
                partBoxes = cell(1, length(parts));
                for j=1:length(parts) % for each ub part                    
                    partJ_X = dpInfo(j).Ix(I2); % indexes in of distance transform grids
                    partJ_Y = dpInfo(j).Iy(I2);
                    [partJ_CX, partJ_CY] = MUB_CfgBox.partCenter(partJ_X, partJ_Y, ...
                        unionScale, xyOffset, rootSize); % the center of the part in im scale
                    
                    partJ_S = dpInfo(j).Is(I2); % scale
                    partJ_SZ = [partJ_S*rootSize(1), partJ_S*rootSize(2)] ; % height & width
                    
                    partJ_ubScore = dpInfo(j).ubScore(I2);                    
                    partBoxes{j} = [partJ_CX - (partJ_SZ(:,2)-1)/2, ...
                                    partJ_CY - (partJ_SZ(:,1)-1)/2, ...
                                    partJ_SZ(:,2), partJ_SZ(:,1), partJ_ubScore];                     
                end
                cfgBoxes{i} = cat(2, cfgScore2, partBoxes{:}, ubUnionBoxes, ubUnionRefIds);
            end;
            cfgBoxes = cat(1, cfgBoxes{:});         
            if strcmpi(smModel.thresh, 'maxOnly') && ~isempty(cfgBoxes)
                [~, maxIdx] = max(cfgBoxes(:,1));
                cfgBoxes = cfgBoxes(maxIdx, :); % return the max only                
            end            
        end

        
        % A special case of Single upper body, or the union is the same with the ub part
        % smModel.ubUnions(:).relBox % 4 numbers for relative location of ubUnion wrt to im43box
        % smModel.ubUnions(:).w % 4 numbers for weights wx, wy, wscale, and wscore
        function cfgBoxes = predict1Ub(smModel, unaryScore)
            imH = unaryScore.imH;
            imW = unaryScore.imW;
            ubScores = unaryScore.scores;
            scales = unaryScore.scales;
            rootSize = unaryScore.rsize;
            xyOffsets = unaryScore.xyOffsets;
            
            
            im43rect = MUB_CfgBox.get43box(imH, imW); % x1, y1, x2, y2
            im43box = [im43rect(1:2), im43rect(3:4) - im43rect(1:2) + 1]; % x1, y1, w, h            
            im43boxH = im43box(4);
            im43boxW = im43box(3);
            
            ubUnions = smModel.ubUnions;            
            nUbUnion = length(ubUnions);
                            
            % NOTE: the smModel MUST have same aspect ratio for all ubUnions!!!
            % i.e., ubUnions(i).relBox(3)/ubUnions(i).relBox(4) is a constant
            ubUnionHeightMul = rootSize(1);            
            ubUnionAspectRatio = 4/3*ubUnions(1).relBox(3)/ubUnions(1).relBox(4); 
            ubUnionWidthMul = ubUnionHeightMul*ubUnionAspectRatio;
            
            % work out the center and height the refUbUnion
            relUbUnions = cat(1, ubUnions(:).relBox);
            refUbUnions = ML_RectUtils.relBoxes2absBoxes(im43box, relUbUnions');
            refUbUnionCenters = refUbUnions(1:2,:) + (refUbUnions(3:4,:)-1)/2;
            refUbUnionHeights = refUbUnions(4,:);
                                    
            nScale = length(scales);   
            cfgBoxes = cell(1, nScale); % output
            
            % a scale level is valid if it contains at least one detection score > thresh
            % This function assuems transformed scores through a logistic function
            % A score of -1 before transformation corresponds to 0.4502 after transformation
            % For cascade detection, a score of -1 is deemed as -inf, and the transformed -inf is 0.
            % In both case, use thresh = 0.4502 is a very safe value.
            % A score of -1 before transformation is already low to not miss any good detection
            % Valid or invalid is only applicable for ubs, not ubUnion
            invalidLevel = false(1, nScale);
            validThresh = 0.4502;
            for i=1:nScale
                invalidLevel(i) = isempty(find(ubScores{i} > validThresh,1));
            end;
            
            % To speed things up, one can skip scales
            for i=1:nScale % for each scale of the ubUnion, each scale can be processed independently                            
                if invalidLevel(i)
                    continue;
                end;
                unionScale = scales(i);
                xyOffset = xyOffsets{i};
                unionUbH = ubUnionHeightMul*unionScale; % height and width of the ubUnion, in image scale
                unionUbW = ubUnionWidthMul*unionScale;
                
                [nFR, nFC] = size(ubScores{i});                
                unaryScore = ubScores{i};
                % Work out the center of ubUnion in the image space
                [X, Y] = meshgrid(1:nFC, 1:nFR);
                % center of the ubUnion
                [CX, CY] = MUB_CfgBox.partCenter(X(:), Y(:), unionScale, xyOffset, rootSize);
                                
                % For each x,y location, find the best reference ubUnion (lowest deformation).
                % The number of nUbUnion is small, don't use repmat, which will be slower                
                cfgScores = zeros(nFC*nFR, nUbUnion);
                for v=1:nUbUnion
                    ubUnion_v = ubUnions(v);
                    wx = ubUnion_v.w(1);
                    wy = ubUnion_v.w(2);
                    ws = ubUnion_v.w(3);
                    wscore = ubUnion_v.w(4);
                    cfgScores(:,v) = wscore*unaryScore(:) ...                    
                        - wx/(im43boxW^2)*(CX - refUbUnionCenters(1, v)).^2 ...
                        - wy/(im43boxH^2)*(CY - refUbUnionCenters(2, v)).^2 ...
                        - ws*(log(unionUbH/refUbUnionHeights(v)))^2;
                end;
                
                [cfgScore, ubUnionRefIds] = max(cfgScores, [], 2);                
                         
                if strcmpi(smModel.thresh, 'maxOnly')
                    [~,I2] = max(cfgScore);                    
                else
                    I2 = find(cfgScore > smModel.thresh);
                end
                
                if isempty(I2)
                    continue;
                end;
                
                CX2 = CX(I2); CY2 = CY(I2);
                ubUnionBoxes = [CX2 - (unionUbW-1)/2, CY2 - (unionUbH-1)/2];
                ubUnionBoxes(:,3) = unionUbW;
                ubUnionBoxes(:,4) = unionUbH;
                ubUnionRefIds = ubUnionRefIds(I2);                                
                cfgScore2 = cfgScore(I2); 
                unaryScore2 = unaryScore(I2);
                cfgBoxes{i} = cat(2, cfgScore2, ubUnionBoxes, unaryScore2, ubUnionBoxes, ubUnionRefIds);
            end;
            cfgBoxes = cat(1, cfgBoxes{:});         
            if strcmpi(smModel.thresh, 'maxOnly') && ~isempty(cfgBoxes)
                [~, maxIdx] = max(cfgBoxes(:,1));
                cfgBoxes = cfgBoxes(maxIdx, :); % return the max only                
            end            
        end

    end
    
    %% Helper functions for training/evaluation
    methods (Static)
        % Build smModel from the centroid of a cluster of configuration features
        % L1C: level-1 centroid 4k*1 vector, for k ubs
        % L2Cs: level-2 centroids 4*m matrix, for m level-2 centroids
        function smModel = crtModel(L1C, L2Cs, stdD)
            L2Cs = L2Cs.*repmat(stdD(1:4), 1, size(L2Cs,2));
            L1C  = L1C.*stdD(5:end);
            
            A = reshape(L1C, 4, length(L1C)/4);            
            A(3:4,:) = exp(A(3:4,:));
            L2Cs(3:4,:) = exp(L2Cs(3:4,:));            
            
            nUbUnion = size(L2Cs, 2);
            for i=1:nUbUnion
                smModel.ubUnions(i).relBox = L2Cs(:,i)';
                smModel.ubUnions(i).w = [100, 100, 100];
            end;
            
            for i=1:size(A,2)
                smModel.parts(i).relBox = A(:, i)';
                smModel.parts(i).w = [100, 100, 100, 1];
            end;
            smModel.thresh = 'maxOnly';
            smModel.type = '234Ub';
        end;
        
        % build smModel for 1 ub case
        % L2Cs: level-2 centroids 4*m matrix, for m level-2 centroids
        function smModel = crtModel1Ub(L2Cs, stdD)
            L2Cs = L2Cs.*repmat(stdD(1:4), 1, size(L2Cs,2));            
            L2Cs(3:4,:)= exp(L2Cs(3:4,:));                        
            nUbUnion = size(L2Cs, 2);
            for i=1:nUbUnion
                smModel.ubUnions(i).relBox = L2Cs(:,i)';
                smModel.ubUnions(i).w = [10000, 10000, 10000, 1];
            end;
                        
            smModel.thresh = 'maxOnly';
            smModel.type = '1Ub';
        end;

        
        % Get the weight vector of a smModel 
        % The weight vector is what we need to train
        function w = getW(smModel)
            if strcmpi(smModel.type, '1Ub')
                weights = cat(2, smModel.ubUnions(:).w);
                unaryW = weights(4:4:end);
                deformW = weights;
                deformW(4:4:end) = [];                            
                w = [unaryW, deformW]';
            else
                weights = cat(2, smModel.parts(:).w);
                unaryW = weights(4:4:end);
                deformW = weights;
                deformW(4:4:end) = [];            
                deformW = cat(2, deformW, smModel.ubUnions(:).w);
                w = [unaryW, deformW]';
            end
        end;
        
        % update the weights of smModel
        % w: weights, a vector of length 3+4k, where k is the number of parts
        %   the first 3*(k+1) numbers are [wx, wy, ws] for the union and the k ubs
        %   the last k numbers are the weights for the detection scores of k ubs.
        function smModel = setW(smModel, w)
            w = w(:)';
            if strcmpi(smModel.type, '1Ub')
                nRefUbUnion = length(smModel.ubUnions);
                if length(w) ~= 4*nRefUbUnion;
                    error('wrong size for the weight vector w');
                end;
                offset = nRefUbUnion;
                for i=1:nRefUbUnion
                    smModel.ubUnions(i).w = [w(offset+3*i-2:offset+3*i), w(i)];
                end;                
            else
                nPart = length(smModel.parts);
                nRefUbUnion = length(smModel.ubUnions);
                
                if length(w) ~= 4*nPart+3*nRefUbUnion;
                    error('wrong size for the weight vector w');
                end;
                
                offset = 4*nPart;
                for i=1:nRefUbUnion
                    smModel.ubUnions(i).w = w(offset+3*i-2:offset+3*i);
                end;
                
                offset = nPart;
                for i=1:nPart
                    smModel.parts(i).w = [w(offset+3*i-2:offset+3*i), w(i)];
                end;
            end
        end;
        
        % Compute multiple cofiguration vectors, one for each refUbUnionId
        % cfgBox: a configuration of ubs, the given refUbUnionId is ignored
        % smModel: a configuration model, only use the relative boxes location are used, not weight
        % cfgVecs: d*k matrix k vectors, one for each reference ub union
        function cfgVecs = cmpMulCfgVecs(cfgBox, smModel, imH, imW)
            nRefUbUnion = length(smModel.ubUnions);
            cfgVecs = cell(1,  nRefUbUnion);
            for refUbUnionId=1:nRefUbUnion
                cfgBox(end) = refUbUnionId;
                cfgVecs{refUbUnionId} = MUB_CfgSM.cmpCfgVec(cfgBox, smModel, imH, imW);
            end;
            cfgVecs = cat(2, cfgVecs{:});
        end;
        
        % Compute a vector of deformation and a vector of ubDet scores 
        % cfgBox: a configuration of ubs        
        % smModel: a configuration model, only use the relative boxes location are used, not weight
        function cfgVec = cmpCfgVec(cfgBox, smModel, imH, imW) 
            im43rect = MUB_CfgBox.get43box(imH, imW); % x1, y1, x2, y2
            im43box = [im43rect(1:2), im43rect(3:4) - im43rect(1:2) + 1]; % x1, y1, w, h            
            im43boxW = im43box(3);
            im43boxH = im43box(4);            
                                    
            % get the ubUnion center
            cfgStruct = MUB_CfgBox.parseCfgBox(cfgBox);
            ubUnion = cfgStruct.ubUnion(1:4);
            ubUnionCeter  = ubUnion(1:2) + (ubUnion(3:4) -1)/2;
            ubUnionHeight = ubUnion(4);
            ubUnionWidth  = ubUnion(3);
            
            refUbUnionId = cfgStruct.ubUnion(5);
                       
            % work out the center and heigh the refUbUnion
            refUbUnion = ML_RectUtils.relBoxes2absBoxes(im43box, ...
                smModel.ubUnions(refUbUnionId).relBox(:));
            refUbUnionCenter = refUbUnion(1:2) + (refUbUnion(3:4)-1)/2;
            refUbUnionHeight = refUbUnion(4);                        
            
            unionFeat = [(ubUnionCeter(1) - refUbUnionCenter(1))^2/(im43boxW^2); ... % normalized by imW (roughly)
                         (ubUnionCeter(2) - refUbUnionCenter(2))^2/(im43boxH^2); ... % normalized by imH
                         log(ubUnionHeight/refUbUnionHeight)^2];

            unionFeatVec = zeros(length(unionFeat), length(smModel.ubUnions));
            unionFeatVec(:, refUbUnionId) = unionFeat; % most but one column is zero.            
            
            if strcmpi(smModel.type, '1Ub')                                
                ubScoreVec = zeros(length(smModel.ubUnions),1);
                ubScoreVec(refUbUnionId) = cfgStruct.ubs(5,:);
                cfgVec = [ubScoreVec; -unionFeatVec(:)];
            else
                nPart = length(smModel.parts);
                feats = cell(1, nPart);
                for i=1:nPart          
                    % reference ub given ubUnion
                    refUb = ML_RectUtils.relBoxes2absBoxes(ubUnion, smModel.parts(i).relBox(:));
                    refUbCenter = refUb(1:2) + (refUb(3:4)-1)/2;
                    refUbHeight = refUb(4);

                    % given ub
                    ub = cfgStruct.ubs(:,i);                
                    ubCenter = ub(1:2) + (ub(3:4) - 1)/2;
                    ubHeight = ub(4);

                    feats{i} = [(ubCenter(1) - refUbCenter(1))^2/ubUnionWidth^2; ...
                                (ubCenter(2) - refUbCenter(2))^2/ubUnionHeight^2; ...
                                log(ubHeight/refUbHeight)^2];                
                end;
                
                deformVec = cat(1, feats{:}, unionFeatVec(:));
                % unary score, i.e., ub detection scores for the constituent ubs
                ubScoreVec = cfgStruct.ubs(5,:);                            
                cfgVec = [ubScoreVec(:); -deformVec];
            end
        end;
                       
        % score include the bias term
        function [score, refUbUnionId, cfgVec] = predict4cfgBox(cfgBox, smModel, imH, imW)
            if ~isnan(cfgBox(end)) % if the refUbUnionId is supplied, use it
                cfgVec = MUB_CfgSM.cmpCfgVec(cfgBox, smModel, imH, imW);
                w = MUB_CfgSM.getW(smModel);
                score = w'*cfgVec + smModel.b; 
                refUbUnionId = cfgBox(end);
            else
                % get all cfgVecs, one per reference ub union
                cfgVecs = MUB_CfgSM.cmpMulCfgVecs(cfgBox, smModel, imH, imW);
                w = MUB_CfgSM.getW(smModel);
                % get the one with highest score
                [maxScore, refUbUnionId] = max(w'*cfgVecs);                
                score = maxScore + smModel.b;
                cfgVec = cfgVecs(:, refUbUnionId);
            end;            
        end;
    end    
end

