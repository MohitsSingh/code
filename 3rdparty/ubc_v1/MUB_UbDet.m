classdef MUB_UbDet
% Code for Ub detection    
% By: Minh Hoai Nguyen (minhhoai@robots.ox.ac.uk)
% Created: 20-Mar-2014
% Last modified: 20-Mar-2014


    methods (Static)
        % Inputs
        %   im: an image
        %   ubDetModel: Dpm model
        %   ubDetCscModel: Dpm cascade model
        %   ubcMmModel: Ubc model
        % Outputs:
        %   ubRects: 5*n for n detected ubs, ubRects(:,i) is left, top, right, bottom, score
        %   isFromUbc: 1*n indicating vector, 
        %       If isFromUbc(i) is 1 if the ub is from the best UBC
        %       If isFromUbc(i) is 0, the ub is from the singleton detector (see the paper)
        function [ubRects, isFromUbc, cfgBox] = ubcCascadeDetect(im, ubDetModel, ubDetCscModel, ubcMmModel)            
            % Look up table for TVHID
            % recall =    [   0.4,    0.5,    0.6,   0.65,    0.7,   0.75,    0.8,     0.85,    0.90]
            %thresholds = [2.2567, 1.6509, 1.0525, 0.7573, 0.3813, 0.0136, -0.3669, -0.8300, -4.1572]; 
            
            unaryScore = MUB_Voc5Cascade.getUnaryScore4im(im, ubDetCscModel, ubDetModel);
            [ubRects, isFromUbc,~,~,cfgBox] = MUB_UbDet.ubcDetectHelper(unaryScore, ubcMmModel);            
        end
        
        % predUbRects: 5*k matrix, predUbRects(:,i) is x1, y1, x2, y2, score
        % isFromUbc: 1*k binary vector, isFromUbc(i)=1 means the i^th detection is the top
        %   detections from UBC. isFromUbc(i)=0 means the detection is added to improve recall.
        function [predUbRects, isFromUbc, bestCmId, bestSmId, cfgBox] = ubcDetectHelper(unaryScore, ubcMmModel, useUbcOnly)
            [svmScore, bestCmId, bestSmId,  cfgBox] = MUB_CfgMM.predict(ubcMmModel, unaryScore);
            if isempty(cfgBox)
                predUbRects = [];
                isFromUbc = [];
                return;
            end;
            cfgBox(1) = svmScore;
            cfgStruct = MUB_CfgBox.parseCfgBox(cfgBox);
            predUbs = cfgStruct.ubs;
            predUbs(5,:) = svmScore;
            predUbRects = ML_RectUtils.boxes2rects(predUbs);
            isFromUbc = true(1, size(predUbRects,2));
            if exist('useUbcOnly', 'var') && ~isempty(useUbcOnly) && useUbcOnly
                return;
            end;
            
            
            % Get additional upper bodies from 1-ub models, to increase recall
            % Use UBC model with 1 ub
            % Don't use UBC models with more than 1 ub, for UBC model with more than 1-ub a high
            % scoring ub is counted many times.
            [cfgScores, cfgBoxes] = MUB_CfgCM.detect(ubcMmModel.cmModels{1}, unaryScore);            
            cfgScores = cfgScores + ubcMmModel.b; % add the bias from mmModel so scores are comparable
            moreUbs = [cfgBoxes(:,2:5), cfgScores]'; % additional from 1-ub models
            thresh = -5; % -1 is too high
            moreUbs(:, moreUbs(5,:) < thresh) = [];


            if ~isempty(moreUbs)
                moreUbRects = ML_RectUtils.boxes2rects(moreUbs);
                % remove one with high overlap with predUbRects
                ub2remove = false(1, size(moreUbRects,2));
                for j=1:size(predUbRects,2)
                    % intersection over union
                    % A threshold of 0.6 slightly decrease the performance
                    overlap = ML_RectUtils.rectOverlap(moreUbRects, predUbRects(:,j));
                    ub2remove(overlap > 0.5) = true;
                    
                    % The next two steps are important. The ub configuraiton brings much
                    % benefit and this emphasizes that we trust the output of UBC detector first.
                    % Without the next two steps, the result will be poor
                    
                    % intersection over area of predUbRect
                    % the result is not sensitive the overlap thershold
                    % I tried from 0.6 to 0.8 and the difference is minimal
                    % The value of 0.7 is best, but only better than other value 0.5%
                    overlap2 = ML_RectUtils.rectOverlap2(moreUbRects, predUbRects(:,j));
                    ub2remove(overlap2 > 0.7) = true;
                    
                    % interesection over area of more rects
                    overlap3 = ML_RectUtils.rectOverlap3(moreUbRects, predUbRects(:,j));
                    ub2remove(overlap3 > 0.7) = true;                    
                end;
                moreUbRects(:, ub2remove) = [];
                
                if ~isempty(moreUbRects)
                    % now do nms among themselves
                    moreUbRects = ML_RectUtils.nms(moreUbRects, 0.5);
                              
                    isFromUbc = cat(2, isFromUbc, false(1, size(moreUbRects,2)));
                    % now concatenate the survial
                    predUbRects = cat(2, predUbRects, moreUbRects);
                end
                
                [predUbRects, pick] = ML_RectUtils.nms(predUbRects, 0.7);
                isFromUbc = isFromUbc(pick);
            end
        end;
        
        % ubRects: return all ubs, no threshold is applied yet
        % Here are lookup table for TVHID:
        % Recall   Prec    Thresh
        %  87.64  32.33   -0.9922
        %  84.75  44.46   -0.8814
        %  80.01  54.84   -0.8116
        %  75.08  65.18   -0.7407
        %  70.15  73.38   -0.6738
        %  67.85  77.19   -0.6412 (thresh for Max F1-score)
        %  60.03  84.93   -0.5679
        % Here are lookup table for SPData+TVHID:
        % Recall   Prec    Thresh
        %  90.01  40.77   -0.8608
        %  85.01  57.96   -0.7560
        %  80.01  70.81   -0.6678
        %  75.01  78.13   -0.6031
        %  70.01  84.41   -0.5423
        %  60.00  90.99   -0.4413
        %  mAp: 81.65
        function ubRects = dpmCascadeDetect(im, ubDetCscModel)
            pyra = featpyramid(im, ubDetCscModel);
            ds = cascade_detect(pyra, ubDetCscModel, -1); % still need to run nms on ds
            ubRects = ds(:,[1:4, 6])';
            ubRects = ML_RectUtils.nms(ubRects, 0.5);            
        end;
    end    
end

