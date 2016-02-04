function masks = getFaceMasks(images,ff_mean)
masks = {};
debug_ = true;
for k = 1:length(images)
    k
    I = images{k};
    sz = size(I);
    img_size = sz(1:2);    

    segments = RemapLabels(vl_slic(im2single(vl_xyz2lab(vl_rgb2xyz(I))),30,1));
    skinprob_n = normalise(computeSkinProbability(double(I)));

    
    g = fspecial('gauss',img_size,mean(img_size/3));
    %      g = fspecial('gauss',img_size,mean(img_size/3));
    if (nargin == 2)
        if (iscell(ff_mean))
            g = im2double(ff_mean{k});
        else
            g = ff_mean;
        end
        skinprob_n  = skinprob_n.*g;
    else
        g  = g/max(g(:));
        skinprob_n  = skinprob_n .* (g.^2);
%          skinprob_n  = skinprob_n .* (g);

    end        
    
    %             
    
    %      skinprob_n  = g;
    killBorders = 10;
    graphData = constructGraph(I,skinprob_n,segments,killBorders);
    [bestL] = applyGraphcut(I,segments,graphData);
    masks{k} = bestL>0;
    
    if (debug_)
        figure(1);
        subplot(1,5,1);imagesc(I); axis image;
        subplot(1,5,2);imagesc(skinprob_n); axis image;
        subplot(1,5,3);imagesc(graphData.seg_probImage); axis image;
        subplot(1,5,4);imagesc(bestL);axis image;
        subplot(1,5,5);imagesc(g);axis image;
        pause
    end
end
end
