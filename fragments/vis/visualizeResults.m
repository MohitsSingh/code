% for
kClass = 20
close all
cls = VOCopts.classes{kClass}
id = 'comp3';
fPath = sprintf(VOCopts.detrespath,id,cls);
[ids,confidence,b1,b2,b3,b4]=textread(sprintf(VOCopts.detrespath,id,cls),'%s %f %f %f %f %f');

[c,ic] = sort(confidence,'descend');
ic = ic(~isnan(c));
ids = ids(ic);
confidence = confidence(ic);
b1 = b1(ic);
b2 = b2(ic);
b3 = b3(ic);
b4 = b4(ic);
%%
% pause;
imgs_so_far = cell(0);

for k = 1:10:length(ic)
    currentID = ids{k};
    a = ismember(imgs_so_far,currentID);
    if (sum(a))
        continue;
    end
    imgs_so_far{end+1} = currentID;
   
    
    imgPath = sprintf(VOCopts.imgpath,currentID);
    I = imread(imgPath);
    clf;
    
    currentScore = confidence(k);
    imshow(I);title(currentScore)
    if (k==1)
        pause;
    end
%     currentScore
    bbox = [b2(k) b1(k) b4(k) b3(k)];        
    hold on;
    plotBoxes2(bbox, 'LineWidth',2,'Color','green');
    pause(.5)
end
