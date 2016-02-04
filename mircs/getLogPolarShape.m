function [x,m_vis] = getLogPolarShape(mask,nTheta,nLayers,m,c)

if (nargin < 2)
    nTheta = 10;
end
if (nargin < 3)
    nLayers = 4;
end

if (nargin < 4)
    m = getLogPolarMask(10,nTheta,nLayers);
end

if (nargin < 5)
    m = imresize(m,size(mask),'nearest');
    u = 1:max(m(:));
     
    c = zeros(length(u),1);
    counts = zeros(size(c));
    
    for q = 1:numel(mask)
        if (m(q))
            c(m(q)) =c(m(q)) + mask(q);
            counts(m(q)) = counts(m(q)) + 1;
        end
    end
    x = c./counts;
    x(counts == 0) = 0;
end
if (nargout == 2)
    u = 1:max(m(:));
    if (nargin < 5)
        m_vis = zeros(size(m));
        for iu = 1:length(u)
            m_vis(m==u(iu)) = c(iu)/counts(iu);
        end
    else
        m_vis = zeros(size(m));
        for iu = 1:length(u)
            m_vis(m==u(iu)) = c(iu);
        end
        x = c;
    end
end
