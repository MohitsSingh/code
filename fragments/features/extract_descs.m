function extract_descs(globalOpts,image_ids,s)
%EXTRACT_DESCS preprocess each of the images in the given list to extract
%descriptors from the tiny images.
%   Detailed explanation goes here
if (nargin < 3)
    s = 0;
end
% Extract PHOW descriptors
for ii = 1:length(image_ids)
    currentID = image_ids{ii};
    fprintf('calculating descriptors for image %s %d/%d [%s%03.3f]\n', ...
        currentID, ii, length(image_ids), '%', 100*ii/length(image_ids));
    descPath = getDescFile(globalOpts,currentID,s);    
    
    if (exist(descPath,'file'))
       delete(descPath);
    end
    imagePath = getImageFile(globalOpts,currentID);
    im = imread(imagePath);
    if (s)
        [F,D] = globalOpts.descfun(im, globalOpts.phowOpts_sample{:}); %#ok<*ASGLU,*NASGU>
    else
        [F,D] = globalOpts.descfun(im, globalOpts.phowOpts{:});
    end
    save(descPath,'F','D');
end