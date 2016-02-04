function sel_ = chooseMulti(Z,Zind)
% allows the user to choose a subset of images from the multiImage Z
finished = false;
figure(1);imshow(Z);title('choose another image (left) or quit (right)');
sel_ = false(1,length(unique(Zind(:))));
LEFT = 1;
RIGHT = 3;
while (~finished)
    [x,y,b] = ginput(1);
    if (b==RIGHT)
        break;
    end
    if (b==LEFT)
        chosenInd = Zind(round(y),round(x));
        sel_(chosenInd) =  ~sel_(chosenInd);
        z = Zind == chosenInd;
        z = cat(3,z,z,z);       
        if (sel_(chosenInd))
            Z(z) = .5*Z(z);
        else
            Z(z) = 2*Z(z);
        end
%          figure(1);
        imshow(Z);
    end
end
close(1);
end