funName = 'unaryStats.m';
[list, ~, ~] = depfun(funName);
% filter out matlab built-in functions
clc
for i=1:length(list)
   if isempty(strfind(list{i},'R2011b'))
      disp(list{i})      
   end
end


% unaryStats_bb
% unaryStats
%