%% Make a legend
% figure
% plot(rand(100,3),'o-');
lh = legend('One','Two','Three','Location','EastOutside');

%% We grab handles 'ch' to the 'Children' of the Legend and use FINDOBJ to
%% find handles 'th' to all the text objects in the legend.
ch = get(lh,'Children');
th = findobj(ch,'Type','text');

%% We grab a handle, 'headerhand', to just the text object containing the
%% string 'Two'. Then we change the font to bold and font size to 14. When we do this, all the text in the legend becomes size 14 and bolded.
headerhand = findobj(th,'String','Two');

%% The text objects are linked to one another using Listeners. We disable
%% (actually, we remove) the listeners using RMAPPDATA so that when we
%% change one of the text objects to be BOLD, the others do not also become
%% BOLD.



for i = 1:length(th)
  if (th(i) == headerhand)
    rmappdata(th(i),'Listeners');
  end
end





set(headerhand,'FontWeight','bold','FontSize',14)
