% d = dir('models/drinking*.mat');
% for k = 1:length(d)
%     oldName = ['models/' d(k).name];
%     newName = strrep(oldName,'drinking','drinking_');%['models/' 'drinking' d(k).name(13:end)];
%     movefile(oldName,newName);
% end