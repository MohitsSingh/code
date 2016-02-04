%%%% experiment 17 %%%%
% 11/12/2013 :-)
% try to estimate facial pose via regression.

initpath;
config;
L_pts = load('/home/amirro/mircs/experiments/experiment_0008/ptsData');
ptsData = L_pts.ptsData(1:2:end);
poses = L_pts.poses(1:2:end);
load('/home/amirro/mircs/experiments/common/faceClusters_big_ims.mat'); % ims
yaw = [poses.yaw];

poseVecs = [[poses.pitch];[poses.yaw];[poses.roll]];

ims_subset = ims(1:2:end);
yaw_subset = col(yaw(1:2:end));

pChns = chnsCompute();
pChns.pColor.enabled =1;
d = cellfun2(@(x) chnsCompute(x,pChns),ims_subset);
dd = cellfun2(@(x) col(cat(3,x.data{:})),d);
dd = cat(2,dd{:});
[ferns,ysPr] = fernsRegTrain(double(dd'),yaw_subset,'loss','exp','eta',.05,...
     'thrr',[0 1],'reg',.01,'S',5,'M',2000,'R',3,'verbose',1);
 
 hist(180*(ysPr-yaw_subset)/pi,30)
 
ims_test = ims(2:2:end);
yaw_test = col(yaw(2:2:end));
d_test = cellfun2(@(x) chnsCompute(x,pChns),ims_test);
dd_test = cellfun2(@(x) col(cat(3,x.data{:})),d_test);
dd_test = cat(2,dd_test{:})';

yaw_pred = fernsRegApply(double(dd_test),ferns);
% mkdir ~/mircs/experiments/experiment_0017/
save ~/mircs/experiments/experiment_0017/ferns.mat ferns

[r,ir] = sort(yaw_pred,'descend');
for k = 1:10:length(ir)
%     if (imageData.test.faceScores(ir(k)) < -.3)
%         continue;
%     end
%     k
% %     break
    clf; imagesc(ims_test{ir(k)});
    title(num2str(r(k)*180/pi));
%     tpause(.1)
    drawnow
end

hist(180*(yaw_pred-yaw_test)/pi,30)

d_ = cellfun2(@(x) chnsCompute(imResample(x,[80 80],'bilinear'),pChns),faces.test_faces);
dd_faces = cellfun2(@(x) col(cat(3,x.data{:})),d_);
dd_faces = cat(2,dd_faces{:});

yaw_pred = fernsRegApply(double(dd_faces)',ferns);

[r,ir] = sort(yaw_pred,'descend');
for k = 1:1:length(ir)
    if (imageData.test.faceScores(ir(k)) < -.7)
        continue;
    end
    if (~imageData.test.labels(ir(k)))
        continue;
    end
    k
%     break
    clf; imagesc(faces.test_faces{ir(k)});
    title(num2str(r(k)*180/pi));
    pause
    drawnow
end





% cluster according to poses....
[IDX_pose,C_pose] = kmeans2(poseVecs',10,struct('nTrial',1,'display',1));

% plot3(poseVecs(1,:),poseVecs(2,:),poseVecs(3,:),'ro')

% edges = linspace(-120,120,30);
% [n,bin] = histc(yaw,edges);
bar(linspace(-120,120,30),n);