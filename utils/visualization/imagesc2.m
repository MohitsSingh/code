function h=imagesc2(I,varargin)
if (ischar(I))
    I = imread(I);
end
h=imagesc(squeeze(I),varargin{:});axis equal; hold on; axis off;% drawnow;
if (length(size(I))==2)
%     colormap gray;
end
end