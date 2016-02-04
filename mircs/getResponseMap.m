function [responses,M,resp_orig] = getResponseMap(conf,imgData,model,mode,sub_f,sub_ff)

if nargin < 4
    mode = 'subImage';
end
responses = [];
if (nargin < 5)
    sub_f = 1.5;
end
if (nargin < 6)
    sub_ff = 100;
end

switch mode
    case 'subImage'
        [M,landmarks,I_rect] = getSubImage(conf,imgData,sub_f);
        if (isempty(M))
            return;
        end
        
        [M,s] = rescaleImage(M,max(sub_ff,sub_ff*sub_f),true);
        
    otherwise
        conf.get_full_image = false;
        [M,I_rect] = getImage(conf,imgData.imageID);
        s = 1;
end

[responses, bs] = imgdetect(M, model,-1.1);%model.thresh);
if (isempty(responses))
    return;
end
responses = responses(:,[1:4 6]);
responses = responses(nms(responses,.8),:); % mild non-maximal suppression.
resp_orig = responses;
responses(:,1:4) = bsxfun(@plus,resp_orig(:,1:4)/s,I_rect([1 2 1 2]));
end