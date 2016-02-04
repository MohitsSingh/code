load train_landmarks_full.mat
load test_landmarks_full.mat
load train_landmarks_full2.mat
load test_landmarks_full2.mat

landmarks_train = {};
for k = 1:length(train_landmarks_full)
    k
    r = train_landmarks_full{k};
    for q = 1:length(r)
        r(q).xy = 2*r(q).xy;
    end
    
    landmarks_train{k} = [r,train_landmarks_full2{k}];
%     landmarks_train{k} = r;
end

landmarks_test = {};
for k = 1:length(test_landmarks_full)
    k
     r = test_landmarks_full{k};
    for q = 1:length(r)
        r(q).xy = 2*r(q).xy;
    end
%     landmarks_test{k}= r;
    landmarks_test{k} = [r,test_landmarks_full2{k}];
end