function SaveSegments(S, strengths, fname)

% Save segments S to fname
%
% Input:
% S(:,id) = [id chain_id center_x center_y orientation length]'
% strengths(id) = corresp strengths
%
% Written file output:
% one line per segment, in format:
% id center_x center_y orientation length strength
%

fid = fopen(fname,'w');

for six = 1:size(S,2)
  l = S(:,six);
  line = [l([1 3:6])' strengths(six)];
  fprintf(fid, '%d %.2f %.2f %.2f %.2f %.3f\n', line);
end

fclose(fid);
