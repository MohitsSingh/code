function [test_scores,test_labels] = to_orig_test(test_scores,fra_db,sel_test)
all_scores = -1000*ones(size(fra_db));
all_scores(sel_test) = test_scores;
test_scores = all_scores(~[fra_db.isTrain]);
test_labels = [fra_db.classID];
test_labels = test_labels(~[fra_db.isTrain]);
test_scores(test_scores==-1000) = min(test_scores(test_scores>-1000));
end
