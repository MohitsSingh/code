function disp_mlnet(mlnet, mlix)

% writes the connections to mlix

if isempty(mlix) return; end

disp(['Connections for ML ' num2str(mlix)]);
disp('Front:');
disp(mlnet(mlix).front);
disp('Back:');
disp(mlnet(mlix).back);
newline; newline;
