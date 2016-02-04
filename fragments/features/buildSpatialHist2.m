function hist = buildSpatialHist(quantImage,bbox,globalOpts)
%BUILDSPATIALHIST Summary of this function goes here
%   Detailed explanation goes here

hists = {};
cc = 0;
normalize_ = 1;
for i = 1:length(globalOpts.numSpatialX)
    
    % simplest first
        
    x_split = bbox(1):(bbox(3)-bbox(1))/globalOpts.numSpatialX(i):bbox(3);
    y_split = bbox(2):(bbox(4)-bbox(2))/globalOpts.numSpatialY(i):bbox(4);
    
    [xx,yy] = meshgrid(x_split,y_split);
    yy = round(yy);
    xx = round(xx);
    %     c = 0;
    %     R = zeros(size(quantImage));    
    
    for row = 1:length(y_split)-1
        for col = 1:length(x_split)-1
            %             c = c+1;
            q = quantImage(yy(row,col):yy(row+1,col+1),...
                xx(row,col):xx(row+1,col+1));
            cc = cc+1;
            %h = accumarray([q(:) ones(numel(q),1)],1,[globalOpts.numWords+1 1]);
            
            nn = globalOpts.numWords+1;
            if (isfield(globalOpts,'map'))
                mm = globalOpts.map;
                mm = mm(mm~=0);
                nn = length(unique(mm))+1;
            end
            
            %h = vl_binsum(zeros(globalOpts.numWords+1,1),1,q(:));
            h = vl_binsum(zeros(nn,1),1,q(:));
            
            %ACCUMULATOR = vl_binsum(ACCUMULATOR,VALUES,INDEXES)
            h = h(1:end-1); % disregard last element, it's for images locations
            
            % not assigned any visual word since they are too close to the
            % edge.
            %             if (cc == 1)
            %             hists{cc} = h/sum(h);
            if (normalize_)
                hists{cc} = h;
            else
                hists{cc} = h/sum(h);
            end
            
            %             else
            %                 hists{cc} = 0*h/sum(h);
            %             end
            
            %             R(yy(ixx,iyy):yy(ixx+1,iyy+1),...
            %                 xx(ixx,iyy):xx(ixx+1,iyy+1)) = c;
        end
    end
end

hist = cat(1,hists{:});
if (normalize_)
    hist = hist / sum(hist) ;
end
%
end
