function e = MLEndpoint(ml, fob)

% coordinates of an endpoint of main-line ml.
% If fob==2 -> front, else back

if fob == 2
  e = [ml(3)+cos(ml(5))*ml(6)/2 ml(4)+sin(ml(5))*ml(6)/2];  % front endpoint
else
  e = [ml(3)-cos(ml(5))*ml(6)/2 ml(4)-sin(ml(5))*ml(6)/2];  % back  endpoint
end
