function classifiers = clusters2classifiers(conf,clusters)
%     class: 'n02823428'
%       weights: [11x7x31 double]
%          bias: 0
%     object_sz: [64 31]
%          name: 'beer bottle'
wsz = [conf.features.winsize 32];
classifiers = struct;
for k = 1:length(clusters)
    classifiers(k).class = '';
    classifiers(k).weights = reshape(clusters(k).w,wsz);
    classifiers(k).bias = clusters(k).b;
    classifiers(k).object_sz = 8*(wsz(1:2));
    classifiers(k).name = '';
end
end