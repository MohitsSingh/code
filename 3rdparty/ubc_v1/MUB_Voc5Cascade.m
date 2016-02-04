classdef MUB_Voc5Cascade
% LSVM VOC-release5 dense detection scores from Cascade detection model
% By: Minh Hoai Nguyen (minhhoai@robots.ox.ac.uk)
% Created: 30-May-2013
% Last modified: 20-Mar-2014
    
    methods (Static)
        % Get dense unary score for a particular image
        % Get a single-component unary score by combining different components
        % This is only valid if detWins and xyOffsets are the same for all components
        % Scores of different components are combined using max
        function [unaryScore, rects] = getUnaryScore4im(im, cscModel, model) 
            [scores, scales, detWins, xyOffsets] = MUB_Voc5Cascade.getDenseDetScores(im, cscModel, model);            
            if nargout > 1
                [unaryScore, rects] = MUB_Voc5Cascade.getUnaryScore(scores, scales, detWins, ...
                    xyOffsets, size(im,1), size(im,2));
            else
                unaryScore = MUB_Voc5Cascade.getUnaryScore(scores, scales, detWins, ...
                    xyOffsets, size(im,1), size(im,2));
            end                
        end;

        
        % Get a single-component unary score by combining different compoents
        % This is only valid if detWins and xyOffsets are the same for all components
        % Scores of different components are combined using max
        function [unaryScore, rects] = getUnaryScore(scores_tmp, scales, detWins_tmp, xyOffsets_tmp, imH, imW)
            detWin = detWins_tmp{1};
            xyOffset = xyOffsets_tmp{1};
            
            nLevel = size(scores_tmp, 2);
            nComp = length(detWins_tmp);
            xyOffsets = cell(1, nLevel);
            
            scores = cell(1, nLevel);
            for level = 1:nLevel
                 xyOffsets{level} = xyOffset;
                 % turn to dense matrix with -inf 
                 for r=1:nComp
                     A = true(size(scores_tmp{r, level}));
                     A(find(scores_tmp{r,level})) = false; %#ok<FNDSB>
                     scores_tmp{r,level} = full(scores_tmp{r,level});
                     scores_tmp{r,level}(A) = -inf;
                 end;
                 
                 scores{level} = max(cat(3, scores_tmp{:,level}), [], 3);
            end;
            
            unaryScore.imH = imH;
            unaryScore.imW = imW;
            unaryScore.scores = MUB_CfgBox.transformScore(scores);
            unaryScore.scales = scales;
            unaryScore.rsize = detWin;
            unaryScore.xyOffsets = xyOffsets;
            
            if nargout > 1 % only return rects with score > -inf
                rects = cell(1, nLevel);
                for level=1:nLevel
                    score = scores{level};
                    scale = scales(level); 
                    
                    I = find(~isinf(score));
                    [Y, X] = ind2sub(size(score), I);
                    [X1, Y1, X2, Y2] = MUB_CfgBox.partRect(X, Y, scale, xyOffsets{level}, detWin);
                    rects{level} = [X1, Y1, X2, Y2, score(I)];
                end;
                rects = MUB_CfgBox.transformRects(rects);
            end;               
        end;

        
        % Get dense detection score from LSVM cascade model
        % scores: nComp*nLevel cell structure, scores{r, l}: a sparse matrix for dense score for 
        %   component r at level l
        % scales: 1*nLevel vector for scales at different levels
        % detWins: 1*nComp cell structure detWins{r} is [detH, detW] for detection window 
        %   of component r
        % xyOffsets: 1*nComp cell structure, xyOffsets{r} is [xOffset, yOffset]
        % rects: nComp*nLevel for rectangles rects{r, l} is n*5 matrix for [X1, Y1, X2, Y2, score]
        %   that corresponds to dense score but with boxes information
        function [scores, scales, detWins, xyOffsets, rects] = getDenseDetScores(im, cscModel, model)
            pyra = featpyramid(im, cscModel);
            ds = cascade_detect(pyra, cscModel, cscModel.thresh); % detection boxes            
                        
            rules = model.rules{model.start};
            nComp = length(rules); % number of components of the detector
            [detWins, xyOffsets] = deal(cell(1, nComp)); 
            
            for r=1:nComp
                detWins{r} = rules(r).detwindow;
                shiftWin = rules(r).shiftwindow;
                xyOffsets{r} = [shiftWin(2) + pyra.padx + 1, shiftWin(1) + pyra.pady + 1];                
            end;
            
            % detect at each scale  
            nInterval = model.interval;
            % skip the first few levels because the scores are -inf
            % this is because parts are at level which is half the scale of the root level
            % For root part at low level, there is no plausible parts so the score is -inf
            nLevel = pyra.num_levels - nInterval;
            [scores, rects] = deal(cell(nComp, nLevel));            
            scales = model.sbin./pyra.scales(1+nInterval:end);   
                        
            for r=1:nComp
                dsr = ds(ds(:,5) == r, :); % detection of component r                
                [X,Y, S] = MUB_CfgBox.reversePartRect(dsr(:,1), dsr(:,2), dsr(:,3), dsr(:,4), ...
                    xyOffsets{r}, detWins{r});
                
                L = round(log(S/scales(1))/log(scales(2)/scales(1))) + 1;
                X = round(X);
                Y = round(Y);
                uniqueL = unique(L);
                for i=1:length(uniqueL)
                    level = uniqueL(i);
                    I_l = L == level;
                    X_l = X(I_l);
                    Y_l = Y(I_l);
                    % h and w might be higher than necessary, but this should not be a problem
                    % but the algorithm using this should take this into account
                    [resH, resW, ~] = size(pyra.feat{level + nInterval}); 
                    scores{r, level} = sparse(Y_l, X_l, dsr(I_l,6), resH, resW);
                    rects{r, level} = dsr(I_l, [1:4, 6]);
                end;
                
                % for the remaining level
                remLs = setdiff(1:nLevel, uniqueL);
                for i=1:length(remLs)
                    level = remLs(i);
                    [resH, resW, ~] = size(pyra.feat{level + nInterval}); 
                    scores{r, level} = sparse(resH, resW);                    
                end;
            end;            
        end
        
        % Get multi-component unary scores for an image
        % mcUnaryScore(i): unaryScore for component i
        % This function is not needed for detecting UBs
        % For UB detection, DPM components have the same aspect ratio (square)
        % So use getUnaryScore4im instead
        function [mcUnaryScore, rects] = getMcUnaryScore4im(im, cscModel, model)
            [scores, scales, detWins, xyOffsets, rects] = MUB_Voc5Cascade.getDenseDetScores(im, cscModel, model);
            
            mcUnaryScore = MUB_Voc5Cascade.getMcUnaryScore(scores, scales, detWins, ...
                xyOffsets, size(im,1), size(im,2));
            rects = MUB_CfgBox.transformRects(rects);            
        end;
        
        % Get multi-component unary scores from dense detection scores
        % mcUnaryScore(i): unaryScore for component i
        % This function is not needed for detecting UBs
        % For UB detection, DPM components have the same aspect ratio (square)
        % So use getUnaryScore4im instead
        function mcUnaryScore = getMcUnaryScore(scores, scales, detWins, xyOffsets, imH, imW)
            xyOffsets_tmp = xyOffsets;
            nLevel = size(scores, 2);
            nComp  = length(detWins);
            xyOffsets = cell(nComp, nLevel);                        
            for comp = 1:nComp
                for level = 1:nLevel
                     score_level = scores{comp, level};                                                                
                     A = true(size(score_level)); % indicator for -inf
                     A(find(score_level)) = false; %#ok<FNDSB>
                     score_level = full(score_level);
                     score_level(A) = -inf;
                     scores{comp, level} = score_level;                     
                     xyOffsets{comp, level} = xyOffsets_tmp{comp};
                end;
            end            
            
            nComp = length(detWins);
            mcUnaryScore = [];
            for i=1:nComp
                mcUnaryScore(i).imH = imH; %#ok<*AGROW>
                mcUnaryScore(i).imW = imW;
                mcUnaryScore(i).scales = scales;
                mcUnaryScore(i).scores = MUB_CfgBox.transformScore(scores(i,:));                
                mcUnaryScore(i).rsize  = detWins{i};
                mcUnaryScore(i).xyOffsets = xyOffsets(i,:);                
            end            
        end;
    end    
end

