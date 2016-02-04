function pose = estimateFacePose(imgs,ferns)
pChns = chnsCompute();
d_test = cellfun2(@(x) chnsCompute(x,pChns),imgs);
dd_test = cellfun2(@(x) col(cat(3,x.data{:})),d_test);
dd_test = cat(2,dd_test{:})';
[pose,poseCM] = fernsRegApply(double(dd_test),ferns);