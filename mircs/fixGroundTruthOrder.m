function groundTruth = fixGroundTruthOrder(groundTruth)
% create a containers.map....

dict = java.util.Hashtable;
c = 0;
lut = zeros(length(groundTruth),1);
for k = 1:length(groundTruth)
    k
    curName = groundTruth(k).sourceImage
    if (isempty(dict.get(curName)))
        c = c+1;
        dict.put(curName,c);
    end
    lut(k) = dict.get(curName);           
end
[s,is] = sort(lut,'ascend');
groundTruth = groundTruth(is);
end