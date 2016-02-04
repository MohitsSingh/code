function curScores = getScores(props)
sel_angle = abs(props(:,1))>30;
curScores = props(:,3);
curScores = curScores+props(:,2);
curScores = curScores+double(props(:,4) > 5);
curScores = curScores+10*props(:,7);
% sel_x = ismember(props(:,5),5:45);
% sel_y = ismember(props(:,6),20:40);
%curScores = curScores .* (sel_x & sel_y & sel_angle);
curScores = curScores .* double(sel_angle);
curScores(isnan(curScores)) = -1;
end