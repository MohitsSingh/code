

classdef CompositeFeatureExtractor < FeatureExtractor
    properties
        extractors
        featureLengths = []
    end
    methods
        function obj = CompositeFeatureExtractor(conf,extractors)
            obj = obj@FeatureExtractor(conf);
            obj.extractors = extractors;
        end
        function x = extractFeaturesHelper(obj,imageID, roi, varargin)
            x = {};
            for k = 1:length(obj.extractors)
                curx = obj.extractors{k}.extractFeatures(imageID,roi, varargin{:});
                %                 curx = bsxfun(@rdivide,curx,sum(curx,1));
                x{k} = curx;
            end
            
            if (isempty(obj.featureLengths))
                obj.featureLengths = cellfun(@(u) size(u,1), x);
            end
            
            x = cat(1,x{:});
            if (obj.doPostProcess)
                x = bsxfun(@rdivide,x,sum(x.^2).^.5);
            end
        end
        
        function x = description(obj)
            x = 'Composite';
        end
        
        % find the length
        function fixNormalization(obj,x)
            % find std and variance of each channel, and set missing values
            % to the channel mean.
            
            x_mean = zeros(size(x(:,1)));
            x_std = zeros(size(x(:,1)));
            edges = [0 cumsum(obj.featureLengths)];
            for k = 1:length(edges)-1
                range_ = edges(k)+1:edges(k+1);
                x_std(range_) = mean(sum(x(range_,:).^2).^.5);
                
                %                 cur_mean = mean(col(x(range_,:)));
                %                 cur_std = std(col(x(range_,:)));
                %                 x_mean(range_) = cur_mean;
                %                 x_std(range_) = cur_std;
            end
            
            % now fix so the mean norm is 1
            %             x_normalized = bsxfun(@rdivide,bsxfun(@minus,x,x_mean),x_std);
            
            %             mean_norm = mean(sum(x_normalized.^2).^.5);
            
            obj.isNormalized = true;
            obj.normalizer(1).mean = x_mean;
            obj.normalizer(1).std = x_std;%*mean_norm;
            
            
            
            
            % % %             x_mean = mean(x,2);
            % % %             x_std = std(x,0,2);
            % % %             edges = [0 cumsum(obj.featureLengths)];
            % % %             for k = 1:length(edges)-1
            % % %                 cur_mean = x_mean(edges(k)+1:edges(k+1));
            % % %                 cur_std = x_std(edges(k)+1:edges(k+1));
            % % %                 bads = cur_std == 0;
            % % %                 std_fix = mean(cur_std(~bads));
            % % %                 mean_fix = mean(cur_mean(~bads));
            % % %                 cur_mean(bads) = mean_fix;
            % % %                 cur_std(bads) = std_fix;
            % % %                 x_mean(edges(k)+1:edges(k+1)) = cur_mean;
            % % %                 x_std(edges(k)+1:edges(k+1)) = cur_std;
            % % %             end
            % % %
            % % %             obj.isNormalized = true;
            % % %             obj.normalizer(1).mean = x_mean;
            % % %             obj.normalizer(1).std = x_std;
        end
    end
end