function [kas kas_ls]= DescribekAS(kas, mls)
    vak=size(kas,1);

% mls(3,4) -> x,y coordinates of center
% mls(5) -> orientation
% mls(6) -> segment length



% Compute a descriptor for each kas (ml1, ml2 , ... , mlvak).
% The descriptor has the following 1+4*nM components (dimensions):
%
% [Rx1 Rx2 ... Rx(vak-1)  Ry1 Ry2 ... Ry(vak-1) theta1 theta2 ... thetavak L1N L2N ...  L2vak  ]
%
% NN : normalisation
% 
% Where mli->ml(i+1) = Ri with orientation Rthetai and RNi length mormalised by NN
% theta* in [0,pi] are the orientations, and L* are the lengths normalized by NN.
% The order ml1, ml2,... is given by the 'top-left-most rule':
% ml1 = leftmost ml if Rx/|R| > 0.2, otherwise ml1 = topmost ml
% 
% the first segment 

[s1 s2]=size(kas);   

% In this case, we need only the orientation
if vak==1
	if isempty(mls)
	kas_ls=[];
	return
	end;
	
    kas_ls=mls([3 4 6],kas);
    kas(2,:)=mod(mls(5,kas),pi);  % orientations without direction, in [0,pi]
else    
    D = [];                     % matrix with all descriptors
    E = [];
    for j = 1:s2
      p=kas(:,j);
      % Which points are the most far from each others ?
      t=PairwiseIPDissimilarity(mls(3:4,p'),mls(3:4,p'));
      NN=max(max(t));

      % Determine order
      % Quadratic ordering
      mlst=mls(3:4,p');  
      
      temp=p';
      for i1=vak:-1:1
          for i2=1:i1-1
              if compare_S(mlst,i2,i2+1)
                  t=mlst(:,i2);mlst(:,i2)=mlst(:,i2+1);mlst(:,i2+1)=t;
                  t=temp(1,i2);
                  temp(1,i2)=temp(1,i2+1);temp(1,i2+1)=t;
              end
          end
      end
      mlst=mls(:,temp);    
      kas(:,j)=temp';
	  
      % Which point is the closest of the barycenter ?
      XYb=sum(mlst(3:4,:),2)/s1;
      t=XYb'*mlst(3:4,:);
      t=mlst(3:4,:)-repmat(XYb,1,s1);
        % iclo is the closest
      [y iclo]=min(sqrt(diag(t'*t)));
      temp=[temp(iclo) temp(setdiff(1:s1,iclo))];
      %kas(:,j)=temp';
      mlst=mls(:,temp);  % We sort again the mls vecteur
      
      
      
      % Vectors length normalisation     
      t=(mlst(3:4,2:s1)-repmat(mlst(3:4,1),1,s1-1))';
      descr=t(:)'/NN;
      
      
      % Vectors orientationsmod(mlst(5,:),pi)
%      theta1 = mls(5,p(1)); theta2 = mls(5,p(2));
%  if theta1 > pi  theta1 = theta1-pi;  end      % orientations without direction, in [0,pi]
%  if theta2 > pi  theta2 = theta2-pi;  end
      an=mlst(5,:);
      an=an.*(an<=pi)+(an-pi).*(an>pi);
      
      descr=[descr an];  % orientations without direction, in [0,pi]
      
      % Length normalisation
      descr=[descr  mlst(6,:)/NN];


      

      if NN == 0
        disp(['Warning: Describekas: NN == 0']);
        disp(['Setting descr to vectors R = [0 0] and segments lengths = [1 1]']);
        descr=[mod(mlst(5,:),pi)  (zeros(1,s1)+1) zeros(1,s1-1) zeros(1,s1-1)]; 
        NN=1;
      end
      
      E = [E [XYb ; NN]];
      D = [D descr'];
    end % loop over input kas

    % Append descriptors to kas (if not there, otherwise automatically overwrite)
    if not(isempty(kas))
      kas(1+s1:1+s1+size(D,1)-1,:) = D;
    end
    % Barycenter coordinates and scale
    kas_ls=E;
    
end

end


function b=compare_S(mls,s1,s2)
   R = mls(1:2,s2) - mls(1:2,s1);
   nR = norm(R);

   if nR> 0 && abs(R(1)/nR) > 0.2     % leftness reliability threshold
     swap = (R(1)<0);
   else
     swap = (R(2)<0);
   end
   b=swap;
end








function old()

 D = [];                     % matrix with all descriptors
    for p = kas
      % determine order
      R = mls(3:4,p(2)) - mls(3:4,p(1));
      nR = norm(R);
      if abs(R(1)/nR) > 0.2     % leftness reliability threshold
        swap = (R(1)<0);
      else
        swap = (R(2)<0);
      end
      if swap
        p = p([2 1]);
        R = -R;
      end

      % compute descriptor components
      theta1 = mls(5,p(1)); theta2 = mls(5,p(2));
      if theta1 > pi  theta1 = theta1-pi;  end      % orientations without direction, in [0,pi]
      if theta2 > pi  theta2 = theta2-pi;  end
      descr = [R/nR; [theta1 theta2]'; mls(6,p(1:2))'/nR];

      if nR == 0
        disp(['Warning: Describekas: nR == 0']);
        disp(['Setting descr to R = [0 0], lengths = [1 1]']);
        descr(1:2) = [0 0]'; descr(5:6) = [1 1]';
        %keyboard;
      end

      D = [D descr];
    end % loop over input kas

    % Append descriptors to kas (if not there, otherwise automatically overwrite)
    if not(isempty(kas))
      kas(3:8,:) = D;
    end
end
