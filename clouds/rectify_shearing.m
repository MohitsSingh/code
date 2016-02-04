
function res = rectify_shearing(H1, H2, imsize)
%     """Compute shearing transform than can be applied after the rectification
%     transform to reduce distortion.
%     See :
%     http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification
%     "Computing rectifying homographies for stereo vision" by Loop & Zhang
%     """
w = imsize(1);
h = imsize(2);

a = [(w-1)/2., 0., 1.]';
b = [w-1., (h-1.)/2., 1.]';
c = [(w-1.)/2., h-1., 1.]';
d = [0., (h-1.)/2., 1.]';

from_homg = @(x) x(1:2)/x(3);

ap = from_homg(H1*a);
bp = from_homg(H1*b);
cp = from_homg(H1*c);
dp = from_homg(H1*d);

x = bp - dp;
y = cp - ap;

k1 = (h*h*x(2)*x(2) + w*w*y(2)*y(2)) / (h*w*(x(2)*y(1) - x(1)*y(2)));
k2 = (h*h*x(1)*x(2) + w*w*y(1)*y(2)) / (h*w*(x(1)*y(2) - x(2)*y(1)));

if k1 < 0
    k1 = -k1;
    k2 = -k2;
end

res = [[k1, k2, 0];[0, 1, 0];[0, 0, 1]];
