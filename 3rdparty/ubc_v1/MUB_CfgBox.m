classdef MUB_CfgBox
% Contain utility functions for CfgBox and beyond    
% By: Minh Hoai Nguyen (minhhoai@robots.ox.ac.uk)
% Created: 16-Apr-2013
% Last modified: 16-Apr-2014

    methods (Static)
        % An interface function for parsing the values of cfgBox
        % cfgBox is returned value of detectUbCfgHelper();
        % cfgBox is a vector with values: total-score, 
        %                                 nUb*[x1, y1, w, h, ubDetScore], (for ubs), 
        %                                 [x1, y1, w, h, ubUnionRefId] (for ubUnion)
        % Output:
        %   cfgStruct.ubs: 5*k matrix for k ubs, ubs(:,i) is x1, y1, w, h, ubDetScore
        %   cfgStruct.score: total score
        %   cfgStruct.ubUnion: 5*1 vector ub union, which are: x1, y1, w, h, refUbUnionId
        function cfgStruct = parseCfgBox(cfgBox)
            cfgStruct.score = cfgBox(1);
            nUb = (length(cfgBox)-1)/5 - 1;            
            cfgStruct.ubs = reshape(cfgBox(2:end-5), 5, nUb);            
            cfgStruct.ubUnion = cfgBox(end-4:end)';
        end;

        
        % cfgBox: a row vector for boxes configuration
        % see detectUbCfgHelper() for the format
        function dispCfgBox(im, cfgBox, prefixStr)
            cfgStruct = MUB_CfgBox.parseCfgBox(cfgBox);
            if ~exist('prefixStr', 'var')
                prefixStr = '';
            end;

            imshow(im);
            % Find the reference box
            im43rect = MUB_CfgBox.get43box(size(im,1), size(im,2)); % x1, y1, x2, y2
            im43box = [im43rect(1:2), im43rect(3:4) - im43rect(1:2) + 1]; % x1, y1, w, h                        
            
            % display the reference box
            rectangle('Position', im43box, 'EdgeColor', 'y', 'LineWidth', 3);                        
            
            % display the ubUnion
            rectangle('Position', cfgStruct.ubUnion(1:4), 'EdgeColor', 'c', 'LineWidth', 3, 'Clipping', 'off');
%             text(cfgStruct.ubUnion(1), cfgStruct.ubUnion(2), ...
%                 sprintf('%s/%d: %.2f', prefixStr, cfgStruct.ubUnion(5), cfgStruct.score), ...
%                 'BackgroundColor',[.7 .9 .7], 'FontSize', 16, ...
%                 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
            text(cfgStruct.ubUnion(1), cfgStruct.ubUnion(2), ...
                sprintf('%s %.2f', prefixStr, cfgStruct.score), ...
                'BackgroundColor',[.7 .9 .7], 'FontSize', 16, ...
                'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
            
            % display the ubs
            %colors = {'c', 'm', 'y'};
            colors = {'r', 'r', 'r'};
            for j=1:size(cfgStruct.ubs,2)
                ub = cfgStruct.ubs(:,j);
                color = mod(j-1, length(colors)) + 1;
                rectangle('Position', ub(1:4), 'EdgeColor', colors{color}, 'LineWidth', 3, 'Clipping', 'off');
                text(ub(1)+ub(3)-1, ub(2)+ub(4)-1, sprintf('%.2f', ub(5)), ...
                    'BackgroundColor',[.7 .9 .7], 'FontSize', 16, ...
                    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
            end;         
        end;

        % Given a set of ubs, find the cfg feature vector, which include deformation and ubDet scores
        % ubs: 4*k for k upper bodies. The order does matter; it should correspond to parts of
        %   a cfgModel although a cfgModel is not an input for this function
        %   ubs(:,i) is x1, y1, w, h
        % refUbUnionId: need to know what refUbUnion the ubs belongs to
        % rects: dense ubDet rects, which are used to find the ubDet scores of the given ubs
        %   the score of ubs are given as the score of rects with highest overlaping ratio
        % rects: 5*n, rects(:,i) is x1, y1, x2, y2, ubDetScore
        function cfgBox = getCfgBox4ubs(ubs, rects) 
            if ~isempty(rects)
                ubRects = ML_RectUtils.boxes2rects(ubs);
                % First, assign the ubDet scores to ubs using the scores of cloest boxes 
                ubDetScore = zeros(1, size(ubs,2));
                adjUbRects = zeros(4, size(ubs,2));
                for i=1:size(ubs,2)
                    overlap = ML_RectUtils.rectOverlap(rects(1:4,:), ubRects(:,i));

                    goodOverlapRects = rects(:, overlap >= 0.5);                
                    if ~isempty(goodOverlapRects)                    
                        [ubDetScore(i), maxIdx] = max(goodOverlapRects(5,:));                     
                        adjUbRects(:,i) = goodOverlapRects(1:4,maxIdx);
                    else                
                        [~,maxIdx] = max(overlap);                
                        ubDetScore(i) = rects(5,maxIdx);                
                        adjUbRects(:,i) = rects(1:4, maxIdx);
                    end
                end;
                adjUbs = ML_RectUtils.rects2boxes(adjUbRects);
                % Find the ubUnion
                ubUnion = ML_RectUtils.getBoxesUnion(adjUbs);            
                adjUbs(5,:) = ubDetScore;            
                cfgBox = [nan, adjUbs(:)', ubUnion', nan]; % the total score is nan coz we don't know
                                                        % the refUbUnionId is nan coz we don't know                                                    
            else
                ubUnion = ML_RectUtils.getBoxesUnion(ubs);            
                ubs(5,:) = 0; % corrrespond to -inf before tranformscore
                cfgBox = [nan, ubs(:)', ubUnion', nan]; % the total score is nan coz we don't know
                                                        % the refUbUnionId is nan coz we don't know                                                                    
            end
        end
        
        % Transform the raw ubDet score to a normalized value, using sigmoid function         
        function score = transformScore(score)
            if iscell(score)
                for i=1:length(score)
                    score{i} = MUB_CfgBox.transformScore(score{i});
                end;                
            else
                score = exp(score/5);
                score = score./(1+score);
            end
        end;
        
        % Inversed transformation from normalized score to raw score
        % see also: transformScore
        function score = inverseTransforeScore(score)
            score = 5*(log(score) - log(1-score));            
        end
        
        % Transform score of rectangles
        function rects = transformRects(rects)
            for i=1:size(rects,1)
                for j=1:size(rects,2) 
                    if ~isempty(rects{i,j})
                        rects{i,j}(:,5) = MUB_CfgBox.transformScore(rects{i,j}(:,5));
                    end
                end
            end;            
        end;

        
        % nearest neighbor resizing of dense score
        % Dense score matrices might have weird width and height so the only way to make them
        % consistent is to work back to the coordiate of the original image.
        % assuming newH <= oldH, newW <= oldW        
        function newMat = resizeDenseScore(mat, oldScale, oldXyOffset, ...
                                            newH, newW, newScale, newXyOffset, ...
                                            detWin)
            % Conceptually clear, but slow code. Use the below
            % Rect center in the image coordinate
            % [X0, Y0] = MUB_CfgBox.partCenter((1:newW)', (1:newH)', newScale, newXyOffset, detWin);            
            % Y, Y in old mat coordinate
            % [X, Y] = MUB_CfgBox.reversePartCenter(X0, Y0, oldScale, oldXyOffset, detWin);
            
            % This is equivalent to the above code but faster
            step = newScale/oldScale;
            X1 = (detWin(2)/2 - newXyOffset(1))*step + (oldXyOffset(1) - detWin(2)/2);
            Y1 = (detWin(1)/2 - newXyOffset(2))*step + (oldXyOffset(2) - detWin(1)/2);            
            X = (X1+step):step:(X1+newW*step);
            Y = (Y1+step):step:(Y1+newH*step); 
            
            X = round(X);
            Y = round(Y);
            
            [oldH, oldW] = size(mat);
            X = max(X, 1);
            X = min(X, oldW);
            Y = max(Y, 1);
            Y = min(Y, oldH);
            newMat = mat(Y, X);
        end;
        
        
        % X,Y: n*1 vectors
        % scale: scale
        % detWin: 1*2 vector for detH*detW of the part
        % xyOffset: 1*2 vector for offset in x, y
        function [X1, Y1, X2, Y2] = partRect(X, Y, scale, xyOffset, detWin)
            X1 = (X- xyOffset(1))*scale + 1;
            Y1 = (Y- xyOffset(2))*scale + 1;
            X2 = X1 + detWin(2)*scale - 1;
            Y2 = Y1 + detWin(1)*scale - 1;
        end;
        
        % Reverse function of partRect
        function [X, Y, Scale] = reversePartRect(X1, Y1, X2, Y2, xyOffset, detWin)
            ScaleX = (X2 - X1 +1)/detWin(2);
            ScaleY = (Y2 - Y1 +1)/detWin(1);
            X = (X1 - 1)./ScaleX + xyOffset(1);
            Y = (Y1 - 1)./ScaleY + xyOffset(2);
            Scale = 0.5*(ScaleX + ScaleY);
        end;
        
        % Return the center of the parts.
        % The center returned by this function is the center of the rect returned by partRect()
        function [X0, Y0] = partCenter(X, Y, scale, xyOffset, detWin)
            X0 = (X- xyOffset(1) + detWin(2)/2)*scale + 0.5;
            Y0 = (Y- xyOffset(2) + detWin(1)/2)*scale + 0.5;            
        end
        
        % reverse function of partCenter
        function [X, Y] = reversePartCenter(X0, Y0, scale, xyOffset, detWin)
            X = (X0 - 0.5)/scale - detWin(2)/2 + xyOffset(1);
            Y = (Y0 - 0.5)/scale - detWin(1)/2 + xyOffset(2);
        end;
        
        
        % turn frmRec to dense unaryScore structure
        function unaryScore = frmRec2unaryScore(frmRec)
            if isfield(frmRec, 'flip') && frmRec.flip
                denseFile = sprintf('%s/%s_flip.mat', frmRec.denseUbDetDir, frmRec.denseUbDetFile);
            else               
                denseFile = sprintf('%s/%s.mat', frmRec.denseUbDetDir, frmRec.denseUbDetFile);                
            end;            
            load(denseFile, 'unaryScore');    
        end;
        
        
        % Given a set of predicted ubs and gt ubs
        % Determine which one are true positive and which are false positive
        % Return the scores for gtUbs (both true positve and false negative) 
        % and scores for false positives
        % This requires Hungrarian algorithm
        function [posScore, negScore, isTruePos] = assgnPosAndNeg(predUbs, gtUbs)
            if isempty(predUbs)
                posScore = -inf(1, size(gtUbs,2));                
                negScore = [];
                isTruePos = [];
            elseif isempty(gtUbs)
                posScore = [];
                negScore = predUbs(5,:); % everything is negative
                isTruePos = false(size(predUbs,2),1);
            else
                predRects = ML_RectUtils.boxes2rects(predUbs);
                gtUbRects = ML_RectUtils.boxes2rects(gtUbs);                
                
                O = zeros(size(gtUbRects,2), size(predRects,2));
                for j=1:size(gtUbRects,2)
                    overlap = ML_RectUtils.rectOverlap(predRects(1:4,:), gtUbRects(1:4,j));
                    O(j,:) = overlap;                    
                end;
                
                O2 = max(predRects(5,:)) - repmat(predRects(5,:), size(gtUbRects,2), 1);
                O2(O < 0.5) = inf; % cannot match to the one with low over
                                   % for the one with significnat overlap, use their score
                                   % for secondary 
                                   % Don't use the overlap amount as it is not reliable
                                
                                                
                % find assignment with max overlap score                
                bestAssgn = assignmentoptimal(O2); % mapping from gt to pred
                posScore = -inf(1, size(gtUbRects,2));
                isTruePos = false(size(predUbs,2),1);
                for j=1:size(gtUbRects,2)
                    predUbId = bestAssgn(j);
                    if bestAssgn(j) == 0 || O(j, predUbId) == 0 % no mapping or mapped to one
                        % with < 0.5 overlap
                        posScore(j) = -inf; % will never be detected                        
                    else
                        posScore(j) = predRects(5, predUbId);
                        isTruePos(predUbId) = true;
                    end;
                end;
                negScore = predRects(5,~isTruePos); % detections which are not positive
                
            end
        end
        
        
        % Compute precision and recall
        % require function ml_precRec
        function [prec, rec, ap, thresh4maxF1, allThreshs] = getApHelper(posDetScores, negDetScores)
            posDetScores = cat(2, posDetScores{:});            
            negDetScores = cat(2, negDetScores{:});
            detScores = cat(2, posDetScores, negDetScores);
            lb = [ones(1, length(posDetScores)), zeros(1, length(negDetScores))];
            [prec, rec, ap, thresh4maxF1, allThreshs] = ml_precRec(detScores(:), lb(:), 0, 0); 
        end
        
        % Given the desire recalls, look up the thresh values and prec        
        % prec: increasing order, rec: decresasing order
        function desThreshs = threshLookup(prec, rec, allThreshs, desireRecs)
            fprintf('Recall   Prec    Thresh\n');
            desThreshs = zeros(length(desireRecs),1);
            for i=1:length(desireRecs)
                desireRec = desireRecs(i);
                idx = find(rec > desireRec, 1, 'last');                
                if isempty(idx)
                    idx = find(~isinf(allThreshs), 1, 'first');
                end;
                desThreshs(i) = allThreshs(idx);                
                fprintf(' %5.2f %6.2f  %8.4f\n', 100*rec(idx), 100*prec(idx), desThreshs(i));
            end;            
        end;
        
        % Get configuration feature vector for verification step
        % The feature vector is concatenation of the relativity b/t
        %   the ubs and the enclosing bbox and the relativity b/t the enclosing bbox and the image.
        %   and also the overlap ratios
        % ubs: 4*2, ubs(:,i) for left, top, width, and height
        %   The order of ubs matters. ubs(:,1) is the ref ub (good), while ubs(:,2) is the new
        %   (unsure) ub
        % This function is to compared with MUB_CfgClust.getCfgFeat        
        function cfgFeat = getCfgFeat4Ver(imH, imW, ubs) 
            imBox = ML_RectUtils.rects2boxes(MUB_CfgBox.get43box(imH, imW)');
            ubUnion = ML_RectUtils.getBoxesUnion(ubs);
            ubs2unionRel = ML_RectUtils.absBoxes2relBoxes(ubUnion, ubs);
            union2imRel  = ML_RectUtils.absBoxes2relBoxes(imBox, ubUnion);
            
            % Take the log because ratio of scales are linear in log-space
            union2imRel(3:4)    = log(union2imRel(3:4));
            ubs2unionRel(3:4,:) = log(ubs2unionRel(3:4,:));
            
            union2imRel(4) = [];
            ubs2unionRel(4,:) = [];
            
            cfgFeat = cat(1, union2imRel(:), ubs2unionRel(:));
            
            ubRects = ML_RectUtils.boxes2rects(ubs);
            o1 = ML_RectUtils.rectOverlap(ubRects(:,1), ubRects(:,2));
            o2 = ML_RectUtils.rectOverlap2(ubRects(:,1), ubRects(:,2));
            o3 = ML_RectUtils.rectOverlap3(ubRects(:,1), ubRects(:,2));
            
            cfgFeat = [cfgFeat; o1; o2; o3];        
        end

        % nearest neighbor resizing
        % assuming newH <= oldH, newW <= oldW    
        % This function is obsolete
        function newMat = matrixResize(mat, newH, newW)
            [oldH, oldW] = size(mat);
            stepX = oldW/newW;
            stepY = oldH/newH;
            IX2 = round(stepX:stepX:oldW);
            IY2 = round(stepY:stepY:oldH);
            newMat = mat(IY2, IX2);            
        end;
        
        % get the bbox which has the height the same as imH
        % but with apsect ratio of 4/3. The box is centered wrt to the image center.
        function bbox = get43box(imH, imW)
            wOffset = (imW - 4*imH/3)/2;
            bbox = [1 + wOffset, 1, imW - wOffset, imH];            
        end;


    end
end

