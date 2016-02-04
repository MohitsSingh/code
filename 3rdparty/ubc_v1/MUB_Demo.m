classdef MUB_Demo
% Demo functions    
% By: Minh Hoai Nguyen (minhhoai@robots.ox.ac.uk)
% Created: 16-Apr-2014
% Last modified: 16-Apr-2014

    properties (Constant)
        ubcModelFile = './models/mmModel_dsc.mat';
        dpmModelFile = './models/ubDet_permuteDsc_nComp-2_nPart-2_cascade.mat'
    end
    
    methods (Static)
        
        % Display detected UBs
        function demo1()            
            im = imread('testIm1.jpg');            
                        
            ubc = load(MUB_Demo.ubcModelFile, 'mmModel'); % load UBC model 
            % 4-UB configurations yield no benefit, remove them to improve detection speed            
            ubc.mmModel.cmModels(4) = []; 
            
            % load DPM model and cascade version to compute dense scores
            dpm = load(MUB_Demo.dpmModelFile, 'cscModel', 'model'); 
            
            % Detect
            ubRects = MUB_UbDet.ubcCascadeDetect(im, dpm.model, dpm.cscModel, ubc.mmModel);            
            
            % thresholding
            threshold = -0.3669; % should be tuned for your dataset
            ubRects = ubRects(:, ubRects(5,:) >= threshold);
            
            imshow(im);
            for i=1:size(ubRects,2)
                rect = ubRects(:,i);
                left   = rect(1);
                right  = rect(3);
                top    = rect(2);
                bottom = rect(4);
                score  = rect(5);
                line([left right right left left], [top top bottom bottom top], ...
                    'LineWidth', 3, 'Color', 'r');
                text(right, bottom, sprintf('%.2f', score), ...
                    'BackgroundColor',[.7 .9 .7], 'FontSize', 16, ...
                    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
            end            
        end;
        
        % display the best configuration only
        function demo2()
            im = imread('testIm1.jpg');
                        
            ubc = load(MUB_Demo.ubcModelFile, 'mmModel'); % load UBC model 
            % 4-UB configurations yield no benefit, remove them to improve detection speed            
            ubc.mmModel.cmModels(4) = []; 
            
            % load DPM model and cascade version to compute dense scores
            dpm = load(MUB_Demo.dpmModelFile, 'cscModel', 'model'); 
            
            % Detect
            [ubRects, isFromUbc, cfgBox] = ...
                MUB_UbDet.ubcCascadeDetect(im, dpm.model, dpm.cscModel, ubc.mmModel);                          %#ok<ASGLU>
            
            % Display the detected configuration
            MUB_CfgBox.dispCfgBox(im, cfgBox, 'UBC:');
        end;

        % display the best configuration + singleton detections
        function demo3()
            im = imread('testIm2.jpg');
                        
            ubc = load(MUB_Demo.ubcModelFile, 'mmModel'); % load UBC model 
            % 4-UB configurations yield no benefit, remove them to improve detection speed            
            ubc.mmModel.cmModels(4) = []; 
            
            % load DPM model and cascade version to compute dense scores
            dpm = load(MUB_Demo.dpmModelFile, 'cscModel', 'model'); 
            
            % Detect
            [ubRects, isFromUbc, cfgBox] = MUB_UbDet.ubcCascadeDetect(im, dpm.model, dpm.cscModel, ubc.mmModel);                         
            
            % Display the detected configuration
            MUB_CfgBox.dispCfgBox(im, cfgBox, 'UBC:');
            
            % get UBs from singleton detection
            ubRects = ubRects(:, ~isFromUbc);
            
            % thresholding
            threshold = -0.3669; % should be tuned for your dataset
            ubRects = ubRects(:, ubRects(5,:) >= threshold);

            % Display singleton detections
            for i=1:size(ubRects,2)
                rect = ubRects(:,i);
                left   = rect(1);
                right  = rect(3);
                top    = rect(2);
                bottom = rect(4);
                score  = rect(5);
                line([left right right left left], [top top bottom bottom top], ...
                    'LineWidth', 3, 'Color', 'g');
                text(right, bottom, sprintf('S:%.2f', score), ...
                    'BackgroundColor',[.7 .9 .7], 'FontSize', 16, ...
                    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
            end             
        end;
        
        function demo4()
            ubc = load(MUB_Demo.ubcModelFile, 'mmModel'); 
            MUB_CfgMM.dispModel(ubc.mmModel);
        end;        
    end    
end

