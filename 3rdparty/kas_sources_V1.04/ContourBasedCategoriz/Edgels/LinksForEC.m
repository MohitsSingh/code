function [curr_links, links_ixs] = LinksForEC(ecix, ept, links)

% Links to/from edelchain ecix at endpoint ept.
%
% Input:
% ept==1 -> start of ec
% ept==2 -> end of ec
%
% Output:
% curr_links(:,i) = [ec ept]'
% links_ixs(i) = index of the link in links
%

links_ix1 = find(links(1,:)==ecix);  
links_ix1 = links_ix1(find(links(3,links_ix1)==ept));
links_ix2 = find(links(2,:)==ecix);  
links_ix2 = links_ix2(find(links(4,links_ix2)==ept));
curr_links = [links([2 4],links_ix1) links([1 3],links_ix2)];
links_ixs = [links_ix1 links_ix2];
