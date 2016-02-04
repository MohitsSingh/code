function visualizeBoxes(globalOpts, bboxes, image_ids)

for kk = 1:size(bboxes,1)    
    currentID = image_ids{kk};
    clf;
    imshow(getImageFile(globalOpts,currentID));        
    hold on;
    plotBoxes2(bboxes(kk,:),'g-','LineWidth',2);
    pause;
end
