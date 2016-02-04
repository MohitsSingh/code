function [sel_train,sel_test,all_labels] = getImageSubset(conf,className,newImageData)
% sel_train contains 1 for positive, 0 for don't care, -1 for negative.
lm = [newImageData.faceLandmarks];
sideViews = abs([lm.c]-7)>=3;
trains = [newImageData.isTrain];
pos = [newImageData.label];
validScores = [newImageData.faceScore]>=-.6;
m = readDrinkingAnnotationFile('train_data_to_read.csv');
objNames = {m.objType};
if (strcmp(className,'all'))
    f = 1:length(objNames);
else
    isClass = @(x) strcmpi(className,x);
    f = find(cellfun(isClass,objNames));
end
m = m(f);
orientations = [m.obj_orientation];

switch className
    case 'bottle'
        subsel_ = abs(abs(orientations) - 90) <= 20;
    case 'cup'
        subsel_ = abs(orientations) <= 45;
    case 'straw'
        subsel_ = 1:length(m);
    otherwise
        subsel_ = 1:length(m);
end
% sel_ = f(subsel_);

imgIndices = cellfun2(@(x) findImageIndex(newImageData,x),{m(subsel_).imageID});
imgIndices = imgIndices(cellfun(@any,imgIndices));imgIndices = [imgIndices{:}];

sub_ = false(size(newImageData)); sub_(imgIndices) = true;
all_labels = double(pos); all_labels(~pos) = -1;
all_labels(pos & trains & ~sub_) = 0; % set non-subclass , yet positive images, to don't care.
sel_train = trains & validScores;
sel_test = ~trains & validScores;
% for k = 1:length(newImageData)
%     if (sel_test(k) & all_labels(k)==1)
%         clf;imagesc(getImage(conf,newImageData(k).imageID)); axis image; pause;
%     end
% end
