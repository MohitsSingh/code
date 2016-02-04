
%%
% quiver(ellipse_centers(req,1),ellipse_centers(req,2),10*ellipse_nn(req,1),10*ellipse_nn(req,2),'y--');

% [g,ig] = sort(goods,'descend');
% clf; imagesc(I_cropped); axis image; hold on;

% hold on; axis image;
% for k = 1:size(ellipses_,1)
%     k
%     iEllipse = ig(k);
%     if (~req(iEllipse))
%         continue;
%     end
%     %     clf; imagesc(I_cropped); axis image; hold on;
%
%     xy = xy_ellipses{iEllipse};
%     %     plot(ellipse_centers(iEllipse,1),ellipse_centers(iEllipse,2),'g-');
%     plot(xy(:,1),xy(:,2),'g-');
%     center_x = ellipses_(iEllipse,2);
%     center_y = ellipses_(iEllipse,1);
%     % % % %
%     % plot the "triangle" between the extreme points and center
%
%     %     plot(center_x,center_y,'m.');
%     %     plot([center_x,xy(1,1)],[center_y,xy(1,2)],'r--');
%     %     plot([center_x,xy(end,1)],[center_y,xy(end,2)],'y--');
%     %     pause;
% end

%% 1. ellipse whose *minor* axis points to the mouth

% for each ellipse plot the direction of the minor axis;
%ellipse_n = ((ellipse_starts-ellipse_centers)+(ellipse_ends-ellipse_centers))/2;
% %
% clf; imagesc(I_cropped); axis image; hold on;
% hold on; axis image;
% for iEllipse = 1:size(ellipses_,1)
%     xy = xy_ellipses{iEllipse};
% %     plot(ellipse_centers(iEllipse,1),ellipse_centers(iEllipse,2),'g-');
%     plot(xy(:,1),xy(:,2),'g-');
%     center_x = ellipses_(iEllipse,2);
%     center_y = ellipses_(iEllipse,1);
% % % % %
%     % plot the "triangle" between the extreme points and center
%
%     plot(center_x,center_y,'m.');
%     plot([center_x,xy(1,1)],[center_y,xy(1,2)],'r--');
%     plot([center_x,xy(end,1)],[center_y,xy(end,2)],'y--');
% end
%

%%

%         function runStateMachine(primitives,G,IG,startNodes,debugInfo)
%
%             global pTypes;
%             % G is the adjacency graph.
%
%             if (isempty(startNodes))
%                 warning('state machine: no candidate ellipses found. aborting');
%                 return;
%             end
%
%             t = 0;
%             n = length(primitives);
%
%             % a state is made of a primitive (or compound primitive) and current
%             % "belief" of the situation.
%             % state names: (note: some states may by mixed...)
%             state_initial = 1; % initial
%             state_first_ellipse = 2;
%             state_ellipse_perp_line = 4;
%             state_ellipse_perp_line_2 = 8;
%             state_accept = 1000;
%             currentState = state_first_ellipse;
%             I = debugInfo.I;
%             S.state = CStack;
%             % S = CList;
%             % S = CStack;
%             S.params = CStack;
%             S.vis = CStack;
%             for k = 1:length(startNodes)
%                 %     state = struct('node',startNodes(k),'state',state_first_ellipse,'extra',[]);
%                 %     addState(S,state,0);
%                 addStateAndNode(S,state_first_ellipse,startNodes(k),struct('parent',0),startNodes(k));
%                 %     state_node_Stack.push([state_first_ellipse startNodes(k)]);
%             end
%
%             while ~S.state.isempty()
%                 [currentState,currentNode,currentParams,currentVis] = getStateAndNode(S);
%
%                 %some visualizations: find neighbors of current state (usually helpful).
%                 N = find(G(currentNode,:));
%                 currentVis = currentVis{1};
%                 clf; imagesc(I); axis image; hold on;
%
%                 for k = 1:length(currentVis)
%                     p_ = primitives{currentVis(k)};
%                     plot(p_.xy(:,1),p_.xy(:,2),'G','LineWidth',2);
%                 end
%
%                 % iterate over neighbors to define new state.
%
%                 %1,p1),l2(p1,p2),l2(p2,p1),l2(p2,p2)),[],3);
%                 %IG : 1-1, 1-2, 2-1, 2-2
%                 p = primitives{currentNode};
%                 ig = IG(currentNode,N);
%                 for iN = 1:length(N)
%                     curNeighbor = N(iN);
%                     % don't allow parents to be neighbors :-/
%                     if (currentParams.parent == curNeighbor)
%                         continue;
%                     end
%                     pN = primitives{curNeighbor};
%                     plot(pN.xy(:,1),pN.xy(:,2),'m','LineWidth',2);
%
%                     % order the points so the adjacent point is in-between
%                     switch(ig(iN))
%                         case 1
%                             p11 = p.endPoint; p12 = p.startPoint;
%                             p21 = pN.startPoint; p22 = pN.endPoint;
%                         case 2
%                             p11 = p.endPoint; p12 = p.startPoint;
%                             p21 = pN.endPoint; p22 = pN.startPoint;
%                         case 3
%                             p11 = p.startPoint; p12 = p.endPoint;
%                             p21 = pN.startPoint; p22 = pN.endPoint;
%                         case 4
%                             p11 = p.startPoint; p12 = p.endPoint;
%                             p21 = pN.endPoint; p22 = pN.startPoint;
%                     end
%
%                     line_S = createLine(p11,p12);
%                     line_T = createLine(p21,p22);
%                     clf; imagesc(I); axis image; hold on;
%                     for k = 1:length(currentVis)
%                         p_ = primitives{currentVis(k)};
%                         plot(p_.xy(:,1),p_.xy(:,2),'G','LineWidth',2);
%                     end
%                     plot(p.xy(:,1),p.xy(:,2),'g','LineWidth',2);
%                     drawEdge(p11,p12);
%                     drawEdge(p21,p22);
%                     plotPoint(p11,'ms','p11');
%                     plotPoint(p12,'ms','p12');
%                     plotPoint(p21,'gs','p21');
%                     plotPoint(p22,'gs','p22');
%
%                     % apply rules, according to states...
%                     switch currentState
%                         case state_first_ellipse
%                             disp('state: looking for perp');
%                             if (pN.typeID == pTypes.TYPE_LINE) %% look for perpendicular line.
%
%                                 T = 180*lineAngle(line_S,line_T)/pi;
%
%                                 % depends if this is a left or right turn; should be
%                                 % consistent with direction of ellipse. If endpoints
%                                 % of ellipse were "flipped", then require the
%                                 % complementray angle.
%                                 turnDirection = 180*angle3Points(p.curveCenter,p11,p12)/pi;
%                                 a = turnDirection < 180;
%                                 if (a && abs(T-80)<20) || (~a && abs(T-270)<20) % next state
%                                     plotPoint(p11,'rs','p11');
%                                     plotPoint(p12,'rs','p12');
%                                     plotPoint(p21,'gs','p21');
%                                     plotPoint(p22,'gs','p22');
%                                     %                         drawEdge(createEdge(p11(1),p12(1),,p12-p11));
%                                     % add the state along with the directed line.
%                                     stateParams = struct;
%                                     stateParams.edge = [p21 p22];
%                                     stateParams.turnDirection = turnDirection;
%                                     stateParams.t = 1;
%                                     stateParams.parent = currentNode;
%                                     addStateAndNode(S,state_ellipse_perp_line,curNeighbor,stateParams,[currentVis curNeighbor]);
%                                 end
%                             end
%
%                         case state_ellipse_perp_line
%                             disp('state: looking for perp 2 or ellipse');
%                             if (pN.typeID == pTypes.TYPE_LINE) %% look for line with same direction
%
%                                 prevPoints = currentParams.edge; % second line must be in same direction,
%                                 line_S = createLine(prevPoints(1:2),prevPoints(3:4));
%                                 % using the far_point-near_point.
%                                 d = l2(prevPoints(3:4),[ pN.startPoint;pN.endPoint]);
%                                 [m,im] = min(d,[],2);
%                                 if (im == 1)
%                                     p21 = pN.startPoint; p22 = pN.endPoint;
%                                 else
%                                     p22 = pN.startPoint; p21 = pN.endPoint;
%                                 end
%                                 line_T = createLine(p21, p22);
%                                 T = 180*lineAngle(line_S,line_T)/pi;
%                                 if (T < 10 || T > 350)
%                                     disp('accepting'); pause;
%                                     continue;
%                                 end
%                             elseif (pN.typeID == pTypes.TYPE_ELLIPSE) % look for ellipse at bottom of line...
%                                 T = 180*lineAngle(line_S,line_T)/pi;
%                                 plot(pN.xy(:,1),pN.xy(:,2));
%                                 prevDirection = currentParams.turnDirection;
%                                 % turning direction should be consistent with previous
%                                 % one.
%
%                                 curTurnDirection = 180*angle3Points(pN.curveCenter,p21,p22)/pi;
%                                 a = turnDirection < 180;
%                                 b = curTurnDirection < 180;
%                                 if (a==b)
%                                     if (a && abs(T-80)<20) || (~a && abs(T-270)<20) % next state
%
%                                         %                     if (abs(T-80)<20)
%                                         plotPoint(p11,'rs','p11');
%                                         plotPoint(p12,'rs','p12');
%                                         plotPoint(p21,'gs','p21');
%                                         plotPoint(p22,'gs','p22');
%
%                                         disp('accepting'); pause;
%                                         continue;
%                                         %                         drawEdge(createEdge(p11(1),p12(1),,p12-p11));
%                                         % add the state along with the directed line
%
%                                     end
%                                 end
%                             end
%
%                             plot(pN.xy(:,1),pN.xy(:,2),'b','LineWidth',2);
%                     end
%                 end
%             end
%



% end