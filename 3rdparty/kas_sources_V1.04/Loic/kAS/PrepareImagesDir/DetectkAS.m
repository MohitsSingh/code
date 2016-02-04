function kas = DetectkAS(net, verbose, vak)

    % Detect all multiple adjacent segments (kAS) in network net.
    %
   
    if nargin < 2
      verbose = false;
    end
    if nargin < 3
      vak = 2;
    end
    
    nb=size(net,2);

    % if vak=1 or 2 we can go faster
    if vak==1
        kas=1:nb;
    else
        if vak==20
            if nargin < 2
              verbose = false;
            end

            pas = [];
            for mlix = 1:size(net,2)
              % Add front connections to pas
              connsF = [];
              if not(isempty(net(mlix).front))
                connsF = unique(net(mlix).front(1,:));
                pas = [pas [ones(1,length(connsF))*mlix; connsF]];
                % remove connected mls from net (avoid twin PAS)
                net(mlix).front = [];
                net = RemoveConnections(mlix, connsF, net);
              end
              %
              % Add back connections to pas
              connsB = [];
              if not(isempty(net(mlix).back))
                connsB = unique(net(mlix).back(1,:));
                connsBnF = setdiff(connsB,connsF); % back connections to segms not connected to front
                pas = [pas [ones(1,length(connsBnF))*mlix; connsBnF]];
                % remove connected mls from net (avoid twin PAS)
                net(mlix).back = [];
                net = RemoveConnections(mlix, connsB, net);
              end
            end

            % remove pas of the form p(1)==p(2)
            % (they originate extremely rarely, when an edgelchain containing only one segment is linked to itself)
            if not(isempty(pas))
              pas = pas(:,not(pas(1,:)==pas(2,:)));
            end
            kas=pas;
        % if vak>2 we can use a general method
        else   
               kas = cell(1,nb);
               for mlix = 1:nb
                   connsF=[];
                  % Add front connections to kas
                  if not(isempty(net(mlix).front))
                    connsF = unique(net(mlix).front(1,:));
                    kas{1,mlix} = [kas{1,mlix} connsF];
                  end

                  % Add back connections to kas
                  if not(isempty(net(mlix).back))
                    connsB = unique(net(mlix).back(1,:));
                    connsBnF = setdiff(connsB,connsF);
                    kas{1,mlix} = [kas{1,mlix} connsBnF];      
                  end
               end

            % Now we can produce the kAS, ie the path of length vak in the network
            % We will process by a classic backtracking
              kas=research(1,zeros(1,vak));
              kas=unique(kas,'rows')';
        end
    end

    if verbose
      disp(['Number of segments    : ' num2str(size(net,2))]);
      disp(['Number of kAS         : ' num2str(size(kas,2))]);
      pps = size(kas,2)/size(net,2);
      disp(['Avg kAS per segment   : ' num2str(pps)]);
    end

    function tab=research(d,temp,p) 
        % d - depth, if 0 we stop
        % temp - the temporary vector
        % Which segments must we explore the next time ?
        if d==1
          rs=1:nb;
        else
          % We can go where we want but we can not cross the previous
          rs=setdiff(kas{1,p},temp);
        end;

        % We can do the exploration
        if d==vak+1 
           % We sort the tab
           tab=sort(temp);
        else
          tab=[];
          for i=rs
            temp(d)=i;
            tab=[tab ; research(d+1,temp,i)];
          end
        end
    end
end
