
function [  ] = crossValTrain( trainLabel,trainVec )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
bestcv = 0;
for log2c = -6:10,
    for log2g = -6:3,
        cmd = ['-w1 10 -t 0 -v 2 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv = svmtrain(trainLabel,trainVec', cmd);
        if (cv >= bestcv),
            bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
        end
        fprintf('(best c=%g, g=%g, rate=%g)\n',bestc, bestg, bestcv);
    end
end

end

