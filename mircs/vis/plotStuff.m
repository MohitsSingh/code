function plotStuff(stuff,varargin)
for k = 1:length(stuff)
    plot(stuff(k).xy(:,1),stuff(k).xy(:,2),varargin{:});
end