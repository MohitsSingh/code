function extract_boxes(globalOpts,image_ids)
%EXTRACT_BOXES preprocess each of the images in the given list to extract boxes
%and tiny images
%   Detailed explanation goes here

tic;

if (~iscell(image_ids))
    preprocess_image(globalOpts,image_ids);
    return;
end

for c = 1:length(image_ids)
    if (toc > 1)
        fprintf('checking image %d : %%%03.3f\n',c,100*c/length(image_ids));
        tic;
    end
    %    r     disp(c);
    preprocess_image(globalOpts,image_ids{c});
end
% end