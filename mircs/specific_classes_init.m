% classes = [conf.class_enum.DRINKING;...
%     conf.class_enum.SMOKING;...
%     conf.class_enum.BLOWING_BUBBLES;...
%     conf.class_enum.BRUSHING_TEETH];


% classes = [conf.class_enum.SMOKING];
% % % classes = [conf.class_enum.TEXTING_MESSAGE;
% % %     conf.class_enum.DRINKING;...
% % %     conf.class_enum.SMOKING;...
% % %     conf.class_enum.BLOWING_BUBBLES;...
% % %     conf.class_enum.BRUSHING_TEETH;...
% % %     conf.class_enum.TAKING_PHOTOS;...
% % %     conf.class_enum.WAVING_HANDS;...
% % %     conf.class_enum.READING;...
% % %     conf.class_enum.WRITING_ON_A_BOOK;...
% % %     conf.class_enum.PHONING;...
% % %     conf.class_enum.POURING_LIQUID;...
% % %     conf.class_enum.USING_A_COMPUTER];

classes = [conf.class_enum.DRINKING;...
    conf.class_enum.SMOKING;...
    conf.class_enum.BLOWING_BUBBLES;...
    conf.class_enum.BRUSHING_TEETH;...    
    conf.class_enum.PHONING];
  
    

classNames = conf.classes(classes);

imageNames={newImageData.imageID};
% [r,ir] = sort(imageNames);
% newImageData = newImageData(ir);
class_labels = zeros(1,length(newImageData));
for iClass = 1:length(classes)
    isClass = strncmp(classNames{iClass},imageNames,length(classNames{iClass}));
    class_labels(isClass) = iClass;
end

% d = dir(fullfile(imgDir,'*.jpg'));

use_manual_faces = true; % use manually labeled facial regions.
useAllNegatives = false; % use faces non-action classes as negatives
if (use_manual_faces)
    isValid = class_labels>0;
else
    isValid = [newImageData.faceScore] >-.6;
end
if (~useAllNegatives)
    isValid = isValid & class_labels > 0;
end
isTrain = [newImageData.isTrain] & isValid;
isTest = ~[newImageData.isTrain] & isValid;

faceActionImageNames = imageNames(isValid);
% save faceActionImageNames faceActionImageNames;
isTrain_ = isTrain; isValid_ = isValid;
isTrain = isTrain(isValid);
class_labels = class_labels(isValid);
validIndices = find(isValid);