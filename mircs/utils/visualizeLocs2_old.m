function [p,p_mask] = visualizeLocs2(conf,ids,locs,height,inflateFactor,add_border,draw_rect)
% visualizeLocs
if (nargin < 6)
    add_border = true;
end

if (nargin < 7)
    draw_rect = false;
end

if (nargin < 5)
    inflateFactor = 1; % default is no inflation
end

if (nargin < 4)
    height = 64;
end

r = 0;

p = {};
all_inds = locs(:,11);
u_inds = unique(all_inds);
loadedImages = cell(1,length(ids));
loaded_rects = zeros(length(ids),4);
q = 0;
for iInd = 1:length(u_inds)
    cur_p = find(all_inds==u_inds(iInd));
    %     cur_p = iInd;
    for k = 1:length(cur_p)
        q = q+1
        id_index = locs(cur_p(k),11);
        imageID = ids{id_index};
        
        if (ischar(imageID))
            II = loadedImages{id_index};
            if (isempty(II))
                [I,xmin,ymin,xmax,ymax] = toImage(conf,getImagePath(conf,imageID),1);
                loadedImages{id_index} = I;
                loaded_rects(id_index,:) = [xmin ymin xmax ymax];
            else
                I = II;
                xmin = loaded_rects(id_index,1);
                ymin = loaded_rects(id_index,2);
                xmax = loaded_rects(id_index,3);
                ymax = loaded_rects(id_index,4);
            end
%             if (locs(k,conf.consts.FLIP))
%                 I = flip_image(I);
%             end
            
            size_ratio = (ymax-ymin+1)/conf.max_image_size;
            rect = inflatebbox(locs(cur_p(k),1:4),inflateFactor);
            if (size_ratio > 1) % loc has been taken from cropped image
                rect = rect*size_ratio;
            end
            
            rect([1 3]) = rect([1 3]) + xmin;
            rect([2 4]) = rect([2 4]) + ymin;
            
        else
            I = imageID;
            rect = locs(cur_p(k),1:4);
        end
%         if (locs(k,conf.consts.FLIP))
%             rect = flip_box(rect,size(I));
%         end
        [I_cropped,I_mask] = cropper(I,round(rect));
        if (draw_rect)
            close all;
            imshow(I);
            hold on;
            plotBoxes2(rect([2 1 4 3]),'Color','g','LineWidth',3);
            if (locs(cur_p(k),conf.consts.FLIP))
                title('flip');
            else
                title('no flip');
            end;
            pause;
        end
        if (locs(cur_p(k),conf.consts.FLIP))
            I_cropped = flip_image(I_cropped);
            %             I_cropped = addBorder(I_cropped,5,[1 1 1]);
        end
%          I_cropped = flip_image(I_cropped);
        I_cropped = myResize(I_cropped,height);
        %     imshow(I_cropped);
        I_mask = imresize(I_mask,[height NaN],'nearest');
        if (add_border)
            if (~isempty(strfind(imageID,conf.classes{conf.class_subset})))
                I_cropped =addBorder(I_cropped,3,[0 1 0]);
            else
                I_cropped =addBorder(I_cropped,3,[1 0 0]);
            end
            I_cropped =addBorder(I_cropped,1,[0 0 0]);
        end
        
        p{cur_p(k)} = I_cropped;
        p_mask{cur_p(k)} = I_mask;
    end
end