function dict = learnDictionary(conf,train_ids,toSave)

dictionaryPath = fullfile(conf.cachedir,['dict' conf.suffix '.mat']);
if (toSave && exist(dictionaryPath,'file'))
    load(dictionaryPath);
    return;
end

tt = train_ids(vl_colsubset(1:length(train_ids),100,'Uniform'));

x1 = {};
A = tt;
for ii = 1:length(A);
    disp(['calculating descriptors for dictionary set: %' num2str(100*ii/length(A))]);
    if (ischar(A{ii}))
        I1 = toImage(conf,getImagePath(conf,A{ii}));
    else
        I1 = A{ii};
    end
    [X1] = allFeatures(conf,I1);
    x1{ii} = single(X1);
end

x1 = cat(2,x1{:});
x1 = vl_colsubset(x1,min(100000,size(x1,2)),'Uniform');
[dict] = vl_kmeans(x1, 1000);
save(dictionaryPath,'dict');
% % for q = 1:size(C,2)
% %     imshow(jettify(HOGpicture(reshape(C(:,q),4,4,[]),20)))
% %     pause;
% % end
% %
% % %%
% % tt = train_ids(train_labels);
% % for r = 1:length(tt)
% %
% %     I = toImage(conf,getImagePath(conf,tt{r}));
% %     [X1,uus1,vvs1,scales1,t] = allFeatures(conf,I);
% %
% %     [ bbs ] = uv2boxes( conf,uus1,vvs1,scales1,t );
% %     %  for kk = 1:5
% %     close all;
% %     D = l2(X1',C');
% %     %      [C, A] = vl_kmeans(X1, 10);
% %     %     D = l2(X1',X1');
% %     [M,IM] = sort(D,2,'ascend');
% %     % find the coding erros.
% %     %     [~,norms] = normalize_vec(X1-C(:,A));
% %     norms = M(:,1);
% %     R = zeros(size(I,1),size(I,2));
% %     for k = 1:length(norms)
% %         b = round(bbs(k,1:4));
% %         b(1) = max(b(1),1);
% %         b(2) = max(b(2),1);
% %         b(3) = min(b(3),size(I,2));
% %         b(4) = min(b(4),size(I,1));
% %         R(b(2):b(4),b(1):b(3)) = max(R(b(2):b(4),b(1):b(3)),norms(k));
% %     end
% %
% %     R = R-min(R(:));
% %     R = R/max(R(:));
% %     mm = mean(R(:));
% %     ss = std(R(:));
% %     R = R-mean(R(:)) > .5*ss;
% %     I = im2double(I);
% %     R = cat(3,R,R,R);
% %     figure,imshow(cat(2,I,R.*I));
% %     title(['from: ' num2str(size(X1,2))]);
% %
% %     pause
% %     %  end
% % end


