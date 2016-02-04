function res = nnDisplay(imgs1,imgs2,INN,knn)
res = {};
knn = min(knn,size(INN,2));
for k = 1:length(imgs1)
        curImg = imgs1{k};        
        curImg = addBorder(curImg,3,[0 0 0]);
        nns = imgs2(INN(k,1:knn));        
        res{k} = multiImage([curImg, nns],false,true);
end
end
%nnDisplay(action_faces,non_action_faces,INN,5);