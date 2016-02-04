classdef RegionSampler < handle
    %RegionSampler Sample rectangular regions according to some rule.
    
    properties
        roi = [];
        minRoiOverlap = .1
        roiType = 'region';
        borders = [];
        minAreaInsideBorders = 1;
        %
        boxSize = [32 32];
        debugInfo = [];
        edgeBoxData = [];
        
    end
    
    properties (Access = private)
        %delta = max(1,floor(boxSize(1)*(1-ovp)/(1+ovp)));
        delta = []
        boxOverlap = .5;
    end
    methods
        function obj = RegionSampler()
            obj.edgeBoxData = [];
            
            %             ovp = obj.boxOverlap;
            %             obj.delta = max(1,floor(obj.boxSize(1)*(1-ovp)/(1+ovp)));
        end
        function s = sampleEdgeBoxes(obj,I)
            if isempty(obj.edgeBoxData)
                model=load('/home/amirro/code/3rdparty/edgeBoxes/models/forest/modelBsds'); model=model.model;
                model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
                %%%%% set up opts for edgeBoxes (see edgeBoxes.m)
                opts = edgeBoxes;
                opts.alpha = .65; % step size of sliding window search
                opts.beta  = .75;     % nms threshold for object proposals
                opts.minScore = .01;  % min score of boxes to detect
                %opts.maxBoxes = 1e4;  % max number of boxes to detect
                opts.maxBoxes = 3000;  % max number of boxes to detect
                opts.minBoxArea = 40;
                obj.edgeBoxData = struct('model',model,'opts',opts);
            end
            s = edgeBoxes(I,obj.edgeBoxData.model,obj.edgeBoxData.opts);
            s(:,3:4)=s(:,3:4)+s(:,1:2);
            s = obj.validateSamples(s);
        end
        
        
        function s = sampleOnLine(obj,pStart,pEnd)
            % sample boxes along a line.
            pStart = pStart(:)'; pEnd = .01+pEnd(:)';
            v = pEnd-pStart;
            % delta is now one-dim
            spacing = obj.delta*(v/norm(v));
            centers = [pStart(1):spacing(1):pEnd(1);...
                pStart(2):spacing(2):pEnd(2)]';
            
            % shift everything so edges are eq. distance to start, end pts
            dShift = (pEnd-centers(end,:))/2;
            centers = centers+repmat(dShift,size(centers,1),1);
            s = inflatebbox(centers,obj.boxSize,'both',true);
            s = obj.validateSamples(s);
        end
        function s = sampleOnImageGrid(obj,I)
            s = obj.sampleOnGrid([1 1 fliplr(size2(I))]);
        end
        function s = sampleOnGrid(obj,box)
            if nargin < 2
                if isempty(obj.borders)
                    error('sampleOnGrid must accept a bounding box, or sampler''s border must be defined');
                else
                    box = obj.borders;
                end
            end
            j = obj.delta;
            s = {};
            wndSize = obj.boxSize;
            [cx,cy] = meshgrid(box(1):j(1):box(3),box(2):j(1):box(4));
            s = inflatebbox([cx(:) cy(:)],wndSize,'both',true);
            s = obj.validateSamples(s);
        end
        
        function obj = setRoiMask(obj,p)
            obj.roi=p;
            obj.roiType = 'region';
        end
        function obj = setRoiBox(obj,b)
            obj.roi = b;
            obj.roiType = 'box';
        end
        
        
        %         function obj = setBorders(obj,b)
        %             if numel(b) > 4
        %                 obj.roi = [1 1 fliplr(size2(b))];
        %             else
        %                 obj.roi = b;
        %             end
        %         end
        function obj = clearRoi(obj)
            obj.roi = [];
        end
        %function obj = set.boxOverlap(obj,ovp)
        function obj = SetBoxOverlap(obj,ovp)
            obj.boxOverlap = ovp;
            obj.updateDelta();
        end
        function obj = set.boxSize(obj,sz)
            if (isscalar(sz))
                sz = [sz sz];
            end
            obj.boxSize = sz;
            obj.updateDelta();
        end
        
        function obj = set.minAreaInsideBorders(obj,v)
            if v <= 0 || v > 1
                error('Region sampler: minAreaInsideBorders must be > 0 and <=1')
            end
            obj.minAreaInsideBorders = v;
        end
    end
    
    methods (Access=private)
        function obj = updateDelta(obj)
            S = obj.boxSize(1);
            ovp = obj.boxOverlap;
            %obj.delta = max(1,floor(S*(1-ovp)/(1+ovp)));
            obj.delta = max(1,S*(1-ovp)/(1+ovp));
        end
        
        function samples = validateSamples(obj,samples)
            z = size(samples,2);
            samples(:,z+1) = 0;
            
            if ~isempty(obj.roi)
                [~,~,a] = BoxSize(samples);
                switch obj.roiType
                    case 'box'
                        [~,ints] = boxesOverlap(samples,obj.roi);
                        samples(ints./a > obj.minRoiOverlap,z+1) = 1;
                    case 'region'
                        
                        centers = round(boxCenters(samples));
                        
                        locs = sub2ind2(size(obj.roi),fliplr(centers));
                        vals = obj.roi(locs);
                        inRoi = vals > obj.minRoiOverlap;
                        dontCare = vals > 0 & ~inRoi;
                        samples(inRoi,z+1) = 1;
                        samples(dontCare,z+1) = 3;
                        
                        %                         boxRegions = {};
                        %                         for t = 1:size(samples,1)
                        %                             boxRegions{t} = box2Region(samples(t,:),obj.roi);
                        %                         end
                        %
                        %                         [~,ints] = regionsOverlap(	,obj.roi);
                        %
                        %                         [r,ir] = sort(ints,'descend');
                        %                         for tt = 1:length(ir)
                        %                             k = ir(tt);
                        %                             displayRegions(obj.debugInfo,boxRegions,ints./a);
                        %                         end
                        
                end
                
            end
            if (~isempty(obj.borders))
                [~,ints] = boxesOverlap(samples,obj.borders);
                [~,~,a] = BoxSize(samples);
                samples(ints./a < obj.minAreaInsideBorders,z+1) = 2;
            end
        end
        
    end
end