function plotSegs(segs,varargin)
    hold on;
    s = {};
    
    for i = 1:size(segs,1)
        s{i} = [segs(i,1) segs(i,2);segs(i,3),segs(i,4);NaN NaN];
    end
    s = cat(1,s{:});
    plot(s(:,1),s(:,2),varargin{:});
end