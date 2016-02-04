function [image_set,gt_labels] = splitImages(conf,imageSet)
[ids,class_labels] = getImageSet(conf,imageSet);
f_true = find(class_labels);
f_true = f_true(1:1:end);
f_false = find(~class_labels);
f_false = f_false(1:end);

ids_true_val = ids(f_true);
ids_false_val = ids(f_false);

image_set = [ids_true_val(1:end);ids_false_val(1:end)];
gt_labels = zeros(1,length(image_set));
gt_labels(1:length(ids_true_val)) = 1;
end