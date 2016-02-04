function [posFeats,negFeats] = splitFeats(all_features,all_labels)
    posFeats = all_features(:,all_labels~=-1);
    negFeats = all_features(:,all_labels==-1);
end
