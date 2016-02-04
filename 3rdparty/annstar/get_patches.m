function patches=get_patches(frames,scales,imgs,img_names,params)
patches=cell(length(imgs),1);
prev_img_name=[];
for i=1:length(imgs)
    img_id=imgs(i);
    img_name=img_names{img_id};
    %     scale=scales(img_id);
    frame=frames(:,img_id);
    if ~isequal(img_name,prev_img_name)
        %in_img=imreadbw(img_name);
        in_img=imread(img_name);
        if (size(in_img,3)>1)
            in_img = rgb2gray(in_img);
        end
        prev_img_name=img_name;
    end
    rect=frame;
    rect(3:4)=rect(3:4)-rect(1:2);
    patch=imcrop(in_img,rect);    
    
    if any(size(patch)>params.RES_ELEMENTS)
        % retain only RES_ELEMENTS resolution elements in each patch
        patch=imresize(patch,params.RES_ELEMENTS,'bilinear');
    end
    
    patches{i}=patch;
end
end
