function boxes = my_quick_detect(conf,I,winsize,w)
conf.features.winsize = winsize;
[X,uus,vvs,scales,t,boxes ] = allFeatures(conf,I,1 );
V = w'*X;
boxes = [boxes(:,1:4),V(:)];
boxes = boxes(nms(boxes,.5),:);

end