function [ Z,Zind,x,y ] = multiImage(p,toRender,rowVec);
%MULTIIMAGE Summary of this function goes here
%   Detailed explanation goes here


for k = 1:length(p)
    if (length(size(p{k}))==2)
        p{k} = repmat(p{k},[1 1 3]);
    end
end

%firstNonEmpty = find(cellfun(@(x)~isempty(x),p),1,'first');
nonEmpty = find(cellfun(@(x)~isempty(x),p));

sizes = cellfun3(@size2, p(nonEmpty), 1);

if (isempty(nonEmpty))
    error('image array must contain at least one non-empty image');
end
%nn = size(p{nonEmpty});% resize all images to fit non-empty image
nn = round(mean(sizes,1));

% % n_cols = round(sqrt(length(p)*nn(1)/nn(2)))
% % n_rows = round(length(p)/n_cols);
n_cols = ceil(sqrt(length(p)));
n_rows = ceil(length(p)/n_cols);
%ceil(sqrt(length(p)));


% [x,y] = meshgrid(1:nn(2):nn(2)*n_cols,1:nn(1):nn(1)*n_rows);
x = zeros(length(p),1);
y =  zeros(length(p),1);

for k = 1:length(p)
    if (isempty(p{k}))
        p{k} = zeros(nn);
    end
end

if (nargin < 2)
    toRender = true;
end
if (iscell(toRender) || toRender(1))
    base=uint8(1-logical(imread('chars.bmp')));
end
if (nargin < 3)
    rowVec = false;
end

if (nargin == 1 || ~rowVec)
    Z = zeros(n_rows*nn(1),n_cols*nn(2),size(p{1},3),'uint8');
%     Z = zeros(n_rows*nn(1),n_cols*nn(2),3,'uint8');
    Zind = zeros(size(Z,1),size(Z,2),'int16');
    k = 0;
    for r = 1:n_rows
        if (k > length(p))
            break;
        end
        for c = 1:n_cols
            k = k+1;
            if (k > length(p))
                break;
            end
            %     %for k = 1:length(p)
            %         [c,r] = ind2sub([n_cols n_rows],k);
            p{k} = im2uint8(p{k});
            curZ = imResample(p{k},[nn(1) nn(2)]);
            if (size(Z,3)==1 && length(size(curZ))==2)
                curZ = repmat(curZ,[1 1 3]);
            end
            if (iscell(toRender) || toRender(1))
                if (isscalar(toRender))
                    curZ =rendertext(curZ,num2str(k),[250 250 210], [1, 1],base);
                elseif (iscell(toRender))                    
                   bb = toRender{k};
                   bb = bb(1:min(length(bb),9));
                    curZ =rendertext(curZ,'BLEND mode',bb,[250 250 210], [1, 1],base);
                else
                    curZ =rendertext(curZ,num2str(toRender(k)),[250 250 210], [1, 1],base);
%                     curZ =rendertext(curZ,'BLEND mode',toRender(k),[250 250 210], [1, 1],base);
                end
            end
             if (size(Z,3)==1)
                 if (size(curZ,3)==3)
                    curZ = rgb2gray(curZ);
                 end
             end
            Z((r-1)*nn(1)+1:r*nn(1),(c-1)*nn(2)+1:c*nn(2),:)=curZ;
            y(k) = (r-1)*nn(1)+1;
            x(k) = (c-1)*nn(2)+1;
            Zind((r-1)*nn(1)+1:r*nn(1),(c-1)*nn(2)+1:c*nn(2),:)= k;
            
        end
    end
else
    Z = [];
    Zind = [];
    for k = 1:length(p)
        if (k == 1)
            Z_ = im2uint8(p{k});
        else
            Z_ = im2uint8(imResample(p{k},[size(Z,1),NaN]));
        end
        if (toRender)
            if (isscalar(toRender))
                Z_ =rendertext(Z_,num2str(k),[250 250 210], [1, 1],base);
            else
                Z_ =rendertext(Z_,num2str(toRender(k)),[250 250 210], [1, 1],base);
            end
        end
        
        Z = cat(2,Z,Z_);
        Zind = cat(2,Zind,k*ones(size(Z_,1),size(Z_,2)));
    end
   
end

