function [ x1,uus1,vvs1 ] = imageSetFeatures( conf,A,sel_)
%IMAGESETFEATURES Summary of this function goes here
%   Detailed explanation goes here
sel1 = [];
if (nargin == 3)
    sel1 = sel_;
end
for ii = 1:length(A)
    disp(['calculating descriptors for first set: %' num2str(100*ii/length(A))]);
    I1 = getImage(conf,A{ii});
    %toImage(conf,getImagePath(conf,A{ii}));
    [X1,uu,vv] = allFeatures(conf,I1);
    if (any(sel1))
        X1 = X1(:,sel1);
    end
    x1{ii} = single(X1);
    uus1{ii} = uu;
    vvs1{ii} = vv;
end
end