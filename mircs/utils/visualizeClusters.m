function [clusters,allImgs,isclass] = visualizeClusters(conf,ids,clusters,varargin)
%     height,add_border,disp_model)
% visualizeclusters adds to each cluster(I) a visualization of it's top k
% detections in the field clusters(I).vis

% if (nargin < 7)
%     mean_image = 0;
% end

ip = inputParser;
ip.addParamValue('disp_model',false,@islogical);
ip.addParamValue('nDetsPerCluster', min(conf.clustering.top_k,5), @isnumeric);
ip.addParamValue('add_border',false,@islogical);
ip.addParamValue('height',64,@isnumeric);
ip.addParamValue('gt_labels',[],@islogical);
ip.addParamValue('interactive',false,@islogical);
ip.parse(varargin{:});
disp_model = ip.Results.disp_model;
add_border = ip.Results.add_border;
interactive = ip.Results.interactive;
gt_labels = ip.Results.gt_labels;
nDetsPerCluster = ip.Results.nDetsPerCluster;
height = ip.Results.height;

x_image = max(eye(height),fliplr(eye(height)));
r = 0;

loadedImages = cell(1,length(ids));
if (isempty(gt_labels))
    gt_labels = zeros(1,10^5); % sort of a hack, but should work
end

for q = 1:length(clusters)
    q
    if(~clusters(q).isvalid)
        %%
        continue;
    end
    locs = clusters(q).cluster_locs;  
    p = {};
    do_flipcropped =1;
    
    for k = 1:min(size(locs,1),nDetsPerCluster)
%         k
        id_index = locs((k),11);
        II = loadedImages{id_index};
        imageID = ids{id_index};
        if(isempty(II))
            currentIsClass = false;
            if (ischar(imageID))
                if (~isempty(strfind(imageID,conf.classes{conf.class_subset})))
                    currentIsClass = true;
                else
                    currentIsClass = false;
                end
                I = getImage(conf,imageID);
            else
                I = imageID;
            end
            loadedImages{id_index} = I;
        else
            I = II;
            if (ischar(imageID))
                if (~isempty(strfind(imageID,conf.classes{conf.class_subset})))
                    currentIsClass = true;
                else
                    currentIsClass = false;
                end
            end
        end
        
        if (interactive)
            figure(1);clf;imshow(I); hold on;
            plotBoxes2(locs(k,[2 1 4 3]),'g','LineWidth',2);
            %         plotBoxes2([xmin ymin xmax ymax],'m','LineWidth',2);
            k
            if (locs(k,conf.consts.FLIP))
                title('flipped');
            else
                title('not flipped');
            end
            pause;
        end
        %
        I_cropped = cropper(imrotate(I,locs(k,13),'bilinear','loose'),round(locs(k,1:4)));
        if (locs(k,conf.consts.FLIP) && do_flipcropped)
            I_cropped = flip_image(I_cropped);
        end
        
        I_cropped = myResize(I_cropped,height);
        
        if (add_border)
            if currentIsClass || gt_labels(id_index)
                I_cropped =addBorder(I_cropped,3,255*[0 1 0]);
            else
                I_cropped =addBorder(I_cropped,3,255*[1 0 0]);
            end
            I_cropped =addBorder(I_cropped,1,[0 0 0]);
        end
        
        p{k} = im2uint8(I_cropped);
        
        r = r+1;
        allImgs{r} = p{k};
        isclass{r} =currentIsClass;
        
    end
    
    
    for kk = k+1:min(nDetsPerCluster,length(p))
        z = zeros(size(p{k}));
        z(:,:,1) = 0;z(:,:,3) = 0;
        z(1:height,1:height,2) = x_image;
        p{kk} =z;
    end
    
    if (disp_model)
        qqq = (jettify(HOGpicture(reshape(clusters(q).w,conf.features.winsize(1),conf.features.winsize(2),[]),20)));
        qqq = imresize(qqq,[height NaN],'bilinear');
        clusters(q).vis = im2uint8(multiImage([{qqq},p],false,false));
    else
        clusters(q).vis = im2uint8(multiImage(p,false,false));
    end
end

