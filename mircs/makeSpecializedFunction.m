% function [chosenInds,otherInds] = makeSpecializedFunction(funName,imgs,ignoreList)
%     ensuredir('specialized');
%     funPath = fullfile('specialized',[funName '.m']));
%     if (~exist(funPath,'file')
%         fid = fopen(funPath,'w+');
%         fclose(fid);
%     end
%
%     otherInds = setdiff(1:length(imgs),ignoreList);
% end
function makeSpecializedFunction(funName)
ensuredir('specialized');
funPath = fullfile('specialized',[funName '.m']);
if (~exist(funPath,'file'))
    fid = fopen(funPath,'w+');
    fclose(fid);
end
end