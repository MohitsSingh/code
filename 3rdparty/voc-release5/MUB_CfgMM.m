classdef MUB_CfgMM
% Three types of models at three levels:
%   smModel: single configuration, multiple locations and scales
%   cmModel: multiple smModels
%   mmModel: multiple cmModels
% This class is for MM model
% By: Minh Hoai Nguyen (minhhoai@robots.ox.ac.uk)
% Created: 06-Apr-2013
% Last modified: 20-Mar-2014

    %% Main functions neede for running the code
    methods (Static)
        % predict: output a single best configuration
        function [bestScore, bestCmId, bestSmId, bestCfgBox] = predict(mmModel, unaryScore)
            nCmModel = length(mmModel.cmModels);
            [cmScores, smIds] = deal(zeros(1, nCmModel));
            cfgBoxes = cell(1, nCmModel);
            for i=1:nCmModel
                [cmScores(i), smIds(i), cfgBoxes{i}] = MUB_CfgCM.predict(mmModel.cmModels{i}, unaryScore);
            end;
            [bestScore, bestCmId] = max(cmScores);
            bestScore = bestScore + mmModel.b;
            bestSmId = smIds(bestCmId);
            bestCfgBox = cfgBoxes{bestCmId};            
        end
        
        % similar to predict, but output all cfgBoxes instead of a single best
        function [topRects, cmIds] = detect(mmModel, unaryScore)
            nCmModel = length(mmModel.cmModels);                            
            boxes = cell(1, nCmModel);
            cmIds = cell(1, nCmModel);
            for i=1:nCmModel
                [cfgScores, cfgBoxes] = MUB_CfgCM.detect(mmModel.cmModels{i}, unaryScore);
                nUb = (size(cfgBoxes, 2)-1)/5 -1;
                ubss = cell(1, nUb);
                for j=1:nUb
                    ubss{j} = [cfgBoxes(:,2+(j-1)*5:5*j), cfgScores];
                end
                boxes{i} = cat(1, ubss{:});
                cmIds{i} = i*ones(1, size(boxes{i},1));
            end;
            boxes = cat(1, boxes{:})';
            cmIds = cat(2, cmIds{:});
            
            rects    = ML_RectUtils.boxes2rects(boxes);
            [topRects, pick] = ML_RectUtils.nms(rects, 0.5);            
            cmIds = cmIds(pick);            
        end;
        
        
        % Display the mmModel on an image
        function dispModel(mmModel)
            nCmModel = length(mmModel.cmModels);
            for i=1:nCmModel
                figure; 
                MUB_CfgCM.dispModel(mmModel.cmModels{i});
            end;            
        end;        
    end
    
    %% Helper functions for training/evaluation, not needed for prediction/testing
    methods (Static)         
        function [prec, rec, ap] = getAp4Model(frmRecs, mmModel, retrievalOpt)
            nTst = length(frmRecs);            
            [posDetScores, negDetScores] = deal(cell(1, nTst));            
                        
            parfor i=1:nTst
                ml_progressBar(i,nTst, 'Testing');
                frmRec = frmRecs(i);
                unaryScore = MUB_CfgBox.frmRec2unaryScore(frmRec);    
                
                predUbs = [];
                if strcmpi(retrievalOpt, 'all') % detect all and run nms
                    topRects = MUB_CfgMM.detect(mmModel, unaryScore);
                    predUbs = ML_RectUtils.rects2boxes(topRects);
                elseif strcmpi(retrievalOpt, 'top') % only predict the best configuration
                    [svmScore, ~, ~, cfgBox] = MUB_CfgMM.predict(mmModel, unaryScore);
                    cfgStruct = MUB_CfgBox.parseCfgBox(cfgBox);
                    predUbs = cfgStruct.ubs;
                    predUbs(5,:) = svmScore;              
                elseif strcmpi(retrievalOpt, 'top+1ub')
                    
                    % first run prediction
                    [svmScore, ~, ~, cfgBox] = MUB_CfgMM.predict(mmModel, unaryScore);
                    cfgStruct = MUB_CfgBox.parseCfgBox(cfgBox);
                    predUbs = cfgStruct.ubs;
                    predUbs(5,:) = svmScore;              
                    
                    % Get additional upper bodies from 1-ub models
                    [cfgScores, cfgBoxes] = MUB_CfgCM.detect(mmModel.cmModels{1}, unaryScore);
                    nUb = (size(cfgBoxes, 2)-1)/5 -1;
                    if nUb ~= 1
                        error('the first cmModel should be for one upper-body');
                    end;
                    
                    cfgScores = cfgScores + mmModel.b; % add the bias from mmModel so scores are comparable                        
                    moreUbs = [cfgBoxes(:,2:5), cfgScores]'; % additional from 1-ub models
                    thresh = -5;
                    moreUbs(:, moreUbs(5,:) < thresh) = [];
                    
                    moreUbRects = ML_RectUtils.boxes2rects(moreUbs);                   
                    predUbRects = ML_RectUtils.boxes2rects(predUbs);
                    
                    % remove one with high overlap with predUbRects
                    ub2remove = false(1, size(moreUbRects,2));
                    for j=1:size(predUbRects,2)
                        overlap = ML_RectUtils.rectOverlap(moreUbRects, predUbRects(:,j));
                        ub2remove(overlap > 0.5) = true;
                    end;
                    moreUbRects(:, ub2remove) = [];
                    
                    % now do nms among themselves
                    moreUbRects = ML_RectUtils.nms(moreUbRects, 0.5);                    
                    moreUbRects(5,:) = moreUbRects(5,:) - 50; % make sure predUbs are selected first
                    
                    moreUbBoxes = ML_RectUtils.rects2boxes(moreUbRects); % convert to box format
                    
                    % now concatenate the survial
                    predUbs = cat(2, predUbs, moreUbBoxes);                    
                end;

                gtUbs = frmRec.ubs;
                if ~isempty(gtUbs)
                    gtUbs = gtUbs(2:5,:);
                    if frmRec.flip
                        gtUbs(1,:) = frmRec.imW - (gtUbs(1,:) + gtUbs(3,:) -1)  + 1;                        
                    end;
                end
                
                [posDetScores{i}, negDetScores{i}] = MUB_CfgBox.assgnPosAndNeg(predUbs, gtUbs);
            end;
            [prec, rec, ap] = MUB_CfgBox.getApHelper(posDetScores, negDetScores);                                    
            plot(rec, prec, 'r'); hold on;             
            axis([0 1 0 1]); axis square;
            fprintf('ap: %g\n', ap);     
        end;
        
        % Get classification accuracy 
        function Conf = getConfMat4Model(mmModel, frmRecs, nUbLb)
            [predScores, predCmIds] = deal(zeros(1,length(frmRecs)));
            
            parfor i=1:length(frmRecs)
                ml_progressBar(i, length(frmRecs));
                frmRec = frmRecs(i);                
                unaryScore = MUB_CfgBox.frmRec2unaryScore(frmRec);
                [svmScore, cmId] = MUB_CfgMM.predict(mmModel, unaryScore);
                predCmIds(i) = cmId;
                predScores(i) = svmScore;                
            end
            predUbLb = predCmIds;
            predUbLb(predScores < 0) = 0;
            
            % compute the confusion matrix
            nMaxUb = max(nUbLb);
            Conf = zeros(nMaxUb+1);
            for i=0:nMaxUb
                for j=0:nMaxUb
                    Conf(j+1,i+1) = sum(and(predUbLb == i, nUbLb == j)); % predict i, but actual lable is j
                end;                
            end;
            disp(Conf);  
            disp(Conf./repmat(sum(Conf,2), 1, nMaxUb+1)*100); % percentage
        end;
        
        % Test the consistency of scores computed at various stages
        function unitTest1()
            load('mmModel.mat');
            
            frmRec = tstFrmRecss{3}(200);  
            unaryScore = MUB_CfgBox.frmRec2unaryScore(frmRec);
            % way 1
            [bestScore1, bestCmId, bestSmId, bestCfgBox] = MUB_CfgMM.predict(mmModel, unaryScore);
            way1 = bestScore1;
            
            % way 2
            cmModel = mmModel.cmModels{bestCmId};
            bestScore2 = MUB_CfgCM.predict(cmModel, unaryScore);
            way2 = bestScore2 + mmModel.b;
            
            % way 3
            w = MUB_CfgCM.getW(cmModel);
            cfgVec = MUB_CfgCM.cmpCfgVec(cmModel, bestSmId, bestCfgBox, frmRec.imH, frmRec.imW);
            way3 = w'*cfgVec + cmModel.b;
            
            
            % way4
            smModel = cmModel.smModels{bestSmId};
            [bestScore4, refUbUnionId] = MUB_CfgSM.predict4cfgBox(bestCfgBox, smModel, frmRec.imH, frmRec.imW);
            way4 = bestScore4 + cmModel.b + mmModel.b;
            
            
            bestCfgBox2 = bestCfgBox;
            bestCfgBox2(end) = nan;
            [bestScore5, refUbUnionId] = MUB_CfgSM.predict4cfgBox(bestCfgBox2, smModel, frmRec.imH, frmRec.imW);
            way5 = bestScore5 + cmModel.b + mmModel.b;
            
            % way 6
            way6 = bestCfgBox(1) + smModel.b + cmModel.b + mmModel.b;
            
            
            fprintf('best score, way1: %g, way2: %g, way3: %g: way4: %g, way5: %g, way6: %g\n', ...
                way1, way2, way3, way4, way5, way6);
            im = imread(frmRec.im);
            if frmRec.flip
                im = flipdim(im, 2);
            end
            subplot(1,2,1); MUB_ShotType.dispFrmRec(frmRec);
            subplot(1,2,2); MUB_CfgBox.dispCfgBox(im, bestCfgBox);
        end;
    end    
end
