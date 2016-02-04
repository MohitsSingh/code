function SavekAS(kas, kas_ls, kas_strengths, fname)

% Save kAS kas to fname
%
% Input:
% as in usual kAS buseness
%
% Written file output:
% one line per kAS, in format:
% segm_id_1 ... segm_id_vak  center_x center_y scale strength  descriptor
%

vak = Value_of_vak(size(kas,1));

fid = fopen(fname,'w');

for kix = 1:size(kas,2)
  line = [kas(1:vak,kix)'  kas_ls(:,kix)'  kas_strengths(kix)  kas((vak+1):end,kix)'];
  fprintf(fid, [repmat('%d ',1,vak) '%.2f %.2f %.2f %.3f' repmat(' %.3f',1,length(line)-vak-4) '\n'], line);
end

fclose(fid);
