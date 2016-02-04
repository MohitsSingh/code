function showMasks(im,masks,toPause)
%SHOWMASKS Summary of this function goes here
%   Detailed explanation goes here
for k = 1:length(masks)
    clf;
    imagesc(im.*repmat(masks{k},[1 1 3]));
    if (nargin > 2)
        if (toPause == 0)
            continue;
        end
        if (toPause ~=-1)
            pause(toPause);
        else
            pause;
        end
    end
end

end

