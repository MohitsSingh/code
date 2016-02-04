classdef MUB_CfgCM
% Three types of models at three levels:
%   smModel: single configuration, multiple locations and scales
%   cmModel: multiple smModels
%   mmModel: multiple cmModels
% This class is for CM model
% By: Minh Hoai Nguyen (minhhoai@robots.ox.ac.uk)
% Created: 06-Apr-2013
% Last modified: 20-Mar-2014
    
    %% Important methods
    methods (Static)
        % Predict: output a single best configuration
        function [bestScore, bestSmId, bestCfgBox, allScores, allCfgBoxes] = ...
                predict(cmModel, unaryScore)
            smModels = cmModel.smModels;                                    
            nModel = length(smModels);            
            allCfgBoxes = cell(1, nModel);
            allScores = zeros(nModel,1);
            for i=1:nModel      
                if strcmpi(smModels{i}.type, '1Ub')
                    cfgBox = MUB_CfgSM.predict1Ub(smModels{i}, unaryScore);
                else
                    cfgBox = MUB_CfgSM.predict(smModels{i}, unaryScore); 
                end
                allCfgBoxes{i} = cfgBox;
                if isempty(cfgBox)
                    allScores(i) = -inf;
                else
                    allScores(i)   = cfgBox(1) + smModels{i}.b;
                end
            end;    
            
            allScores = allScores + cmModel.b;            
            [bestScore, bestSmId] = max(allScores);
            bestCfgBox = allCfgBoxes{bestSmId};        
        end
        
        % Detect: output multiple cfg candiates
        function [scores, cfgBoxes] = detect(cmModel, unaryScore) 
            smModels = cmModel.smModels;
            nSmModel = length(smModels);
            [scores, cfgBoxes] = deal(cell(1, nSmModel));
            for i=1:nSmModel    
                smModels{i}.thresh = -inf;
                if strcmpi(smModels{i}.type, '1Ub')
                    cfgBoxes{i} = MUB_CfgSM.predict1Ub(smModels{i}, unaryScore); 
                else
                    cfgBoxes{i} = MUB_CfgSM.predict(smModels{i}, unaryScore); 
                end
                if ~isempty(cfgBoxes{i})
                    scores{i} = cfgBoxes{i}(:,1) + smModels{i}.b;
                else
                    scores{i} = [];
                end;
            end;    
            scores = cat(1, scores{:});
            scores = scores + cmModel.b;
            cfgBoxes = cat(1, cfgBoxes{:});                
        end;

        % Display the cmModel
        function dispModel(cmModel)            
            if strcmpi(cmModel.smModels{1}.type, '1Ub')
                nUnion = length(cmModel.smModels{1}.ubUnions);
                nC = ceil(sqrt(nUnion));
                nR = ceil(nUnion/nC); 
                for u=1:nUnion
                    subplot(nR, nC, u);                     
                    pad = 50;
                    im = 255*ones(240,320+2*pad,3);
                    imshow(im);
                    im43box = [1+pad,1,320, 240];                    
                    rectangle('Position', im43box, 'EdgeColor', 'b', 'LineWidth', 3);                    
                    ubUnion = ML_RectUtils.relBoxes2absBoxes(im43box, cmModel.smModels{1}.ubUnions(u).relBox');
                    rectangle('Position', ubUnion, 'EdgeColor', 'r', 'LineWidth', 3);
                end
            else                
                nSmModel = length(cmModel.smModels);
                nR = nSmModel;
                nC = length(cmModel.smModels{1}.ubUnions);  
                
                for u=1:nSmModel
                    nUnion = length(cmModel.smModels{u}.ubUnions);
                    for v=1:nUnion
                        subplot(nR, nC, (u-1)*nC+ v);                     
                        pad = 50;
                        im = 255*ones(240,320+2*pad,3);
                        imshow(im);
                        im43box = [1+pad,1,320, 240];
                        rectangle('Position', im43box, 'EdgeColor', 'b', 'LineWidth', 3);                    
                        ubUnion = ML_RectUtils.relBoxes2absBoxes(im43box, cmModel.smModels{u}.ubUnions(v).relBox');
                        rectangle('Position', ubUnion, 'EdgeColor', 'k', 'LineWidth', 3);
                                                
                        for i=1:length(cmModel.smModels{u}.parts)
                            ub = ML_RectUtils.relBoxes2absBoxes(ubUnion, cmModel.smModels{u}.parts(i).relBox');                            
                            rectangle('Position', ub, 'EdgeColor', 'r', 'LineWidth', 3);
                        end
                        
                    end
                end
            end;
            
        end
    end

    %% Less important helper functions, only needed for training/evaluation
    methods (Static)
        % Get the weight vector for the model
        % The weight vector is what we need to train
        function [w, biasInd] = getW(cmModel)
            nSmModel = length(cmModel.smModels);                        
            Ws = cell(1, nSmModel);
            
            for j=1:nSmModel
                w_j = MUB_CfgSM.getW(cmModel.smModels{j});
                Ws{j} = [w_j; cmModel.smModels{j}.b]; % include the bias term
            end;
            w = cat(1, Ws{:});
            d = size(Ws{1},1);
            biasInd = false(d*nSmModel,1);
            biasInd(d:d:end) = true;
        end;
        
        % Set the weight vector of a model
        function cmModel = setW(cmModel, w)
            nSmModel = length(cmModel.smModels);
            d = length(w);
            d2 = d/nSmModel;
            Ws = reshape(w, d2, nSmModel);            
            for j=1:nSmModel
                cmModel.smModels{j}   = MUB_CfgSM.setW(cmModel.smModels{j}, Ws(1:end-1,j));
                cmModel.smModels{j}.b = Ws(end,j);
            end;
        end;


        % get a configuration vector for a cmModel
        function cfgVec = cmpCfgVec(cmModel, smModelId, cfgBox, imH, imW)            
            cfgVec_comp = MUB_CfgSM.cmpCfgVec(cfgBox, cmModel.smModels{smModelId}, imH, imW);
            d = length(cfgVec_comp);
            nSmModel = length(cmModel.smModels);
            cfgVecs = zeros(d+1, nSmModel); % everything is zero except the one for smModelId
            cfgVecs(:, smModelId) = [cfgVec_comp; 1]; % 1 is for the bias term
            cfgVec = cfgVecs(:);
        end;

        % Get score for a particular configuration box
        function [score, smModelId, refUbUnionId] = predict4cfgBox(cfgBox, cmModel, imH, imW)
            smModels = cmModel.smModels;            
            nSmModel = length(smModels);            
            [scores, refUbUnionIds] = deal(zeros(1, nSmModel));
            for i=1:nSmModel
                [scores(i), refUbUnionIds(i)] = MUB_CfgSM.predict4cfgBox(cfgBox, smModels{i}, imH, imW);
            end;
            [score, smModelId] = max(scores);
            refUbUnionId = refUbUnionIds(smModelId);
        end;
        
        function [prec, rec, ap] = getAp4Model(frmRecs, cmModel)
            nTst = length(frmRecs);            
            [posDetScores, negDetScores] = deal(cell(1, nTst));            
                        
            parfor i=1:nTst
                ml_progressBar(i,nTst, 'Testing');
                frmRec = frmRecs(i);
                unaryScore = MUB_CfgBox.frmRec2unaryScore(frmRec);                
                [svmScore, ~, cfgBox] = MUB_CfgCM.predict(cmModel, unaryScore);
                
                cfgStruct = MUB_CfgBox.parseCfgBox(cfgBox);
                predUbs = cfgStruct.ubs;
                predUbs(5,:) = svmScore;
                
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
    end    
end

