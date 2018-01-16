function [node_out,elem_out] = bisect3(node,elem,markedElem)
%% BISECT3 bisect a 3-D triangulation.
%
% [node,elem] = BISECT3(node,elem,markedElem) refines the mesh (node,elem) 
% by bisecting marked elements (markedElem) and minimal neighboring 
% elements to get a conforming and shape regular triangulation. The longest
% edge bisection is implemented. markedElem is a vector containing the
% indices of elements to be bisected. It could be a logical vector of
% length size(elem,1). 
% 
% [node,elem,bdFlag] = BISECT3(node,elem,markedElem,bdFlag) updates
% the boundry condition for boundary faces (bdFlag). It will be used for
% PDEs with mixed boundary conditions.
%
% [node,elem,bdFlag,HB] = BISECT3(node,elem,markedElem,bdFlag,HB) updates the
% hierarchical structure (HB) of the nodes. HB(:,1) are the global indices
% of new added nodes, and HB(:,2:3) the global indices of two parent nodes
% of new added nodes. HB is usful for the interpolation between two grids;
% see also nodeinterpolate. In 3-D, HB is indispensable for MultiGrid
% methods; see mg3. 
%   
% Example
%
%     node = [-1,-1,-1; 1,-1,-1; 1,1,-1; -1,1,-1; -1,-1,1; 1,-1,1; 1,1,1; -1,1,1]; 
%     elem = [1,2,3,7; 1,6,2,7; 1,5,6,7; 1,8,5,7; 1,4,8,7; 1,3,4,7];
%     elem = label3(node,elem);
%     figure(1); subplot(1,3,1); 
%     set(gcf,'Units','normal'); set(gcf,'Position',[0.25,0.25,0.5,0.3]);
%     showmesh3(node,elem,[],'FaceAlpha',0.35); view([210 8]);
%     [node,elem] = bisect3(node,elem,'all');
%     figure(1); subplot(1,3,2);
%     showmesh3(node,elem,[],'FaceAlpha',0.35); view([210 8]);
%     bdFlag = setboundary3(node,elem,'Dirichlet');
%     [node,elem,~,bdFlag] = bisect3(node,elem,[1 3 4],[],bdFlag);
%     figure(1); subplot(1,3,3);
%     showmesh3(node,elem,[],'FaceAlpha',0.35); view([210 8]);
%
% See also bisect, coarsen, coarsen3, nodeinterpolation, mg3.
%
% Reference page in Help browser
% <a href="matlab:ifem meshdoc">ifem meshdoc</a>
% <a href="matlab:ifem bisect3doc">ifem bisect3doc</a> 
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details. 

%% Set up
if isempty(markedElem), return; end
if strcmp(markedElem,'all'), markedElem = (1:size(elem,1))'; end
if islogical(markedElem), markedElem = find(markedElem); end

%% Pre-allocation
N = size(node,1); NT = size(elem,1);
node(N+8*N,1) = 0; 
elem(NT+3*NT,1) = 0; 
generation = zeros(N+6*NT,1,'uint8'); 

%% Local Refinement
%* Find new cutedges and new nodes
%* Bisect all marked elements
%* Find non-conforming elements
%* Update generation of nodes

cutEdge = zeros(8*N,3);        % cut edges
nCut = 0;                       % number of cut edges
nonConforming = true(8*N,1);   % flag of the non-conformity of edges
while ~isempty(markedElem)
    % switch element nodes such that elem(t,1:2) is the longest edge of t
    elem = label3(node,elem,markedElem);
    p1 = elem(markedElem,1);    
    p2 = elem(markedElem,2);
    p3 = elem(markedElem,3); 
    p4 = elem(markedElem,4);
   
   %% Find new cut edges and new nodes
    nMarked = length(markedElem);     % number of marked elements
    p5 = zeros(nMarked,1);            % initialize new nodes
    if nCut == 0                      % If it is the first round, all marked
        idx = (1:nMarked)';           % elements introduce new cut edges;
    else                              % otherwise find existing cut edges
        ncEdge = find(nonConforming(1:nCut)); % all non-conforming edges
        nv2v = sparse([cutEdge(ncEdge,3);cutEdge(ncEdge,3)],...
                      [cutEdge(ncEdge,1);cutEdge(ncEdge,2)],1,N,N);
        [i,j] = find(nv2v(:,p1).*nv2v(:,p2));
        p5(j) = i;
        idx = find(p5==0); 
    end
    if ~isempty(idx)                   % add new cut edges
        elemCutEdge = sort([p1(idx) p2(idx)],2); % all new cut edges
        % find(sparse) eliminates possible duplications in elemCutEdge
        [i,j] = find(sparse(elemCutEdge(:,1),elemCutEdge(:,2),1,N,N));
        nNew = length(i);              % number of new cut edges
        newCutEdge = nCut+1:nCut+nNew; % indices of new cut edges
        cutEdge(newCutEdge,1) = i;     % add cut edges
        cutEdge(newCutEdge,2) = j;     % cutEdge(:,1:2) two end nodes 
        cutEdge(newCutEdge,3) = N+1:N+nNew; % cutEdge(:,3) middle point
        node(N+1:N+nNew,:) = (node(i,:) + node(j,:))/2; % add new nodes
        nCut = nCut + nNew;            % update number of cut edges
        N = N + nNew;                  % update number of nodes
        % incidence matrix of new vertices and old vertices
        nv2v = sparse([cutEdge(newCutEdge,3);cutEdge(newCutEdge,3)],...
                      [cutEdge(newCutEdge,1);cutEdge(newCutEdge,2)],1,N,N);
        [i,j] = find(nv2v(:,p1).*nv2v(:,p2)); % find middle points locally      
        p5(j) = i;
    end
    clear nv2v elemCutEdge
    
    %% Bisect marked elements
    idx = (generation(p5) == 0);
    if length(find(idx)) == 1
        elemGeneration = max(transpose(generation(elem(markedElem(idx),:))));
    else
        elemGeneration = max(generation(elem(markedElem(idx),1:4)),[],2);
    end
    generation(p5(idx)) = elemGeneration + 1;
    elem(markedElem,1:4) = [p4 p1 p3 p5];
    elem(NT+1:NT+nMarked,1:4) = [p3 p2 p4 p5];
    NT = NT + nMarked;
    clear elemGeneration p1 p2 p3 p4 p5
    
    %% Find non-conforming elements
    checkEdge = find(nonConforming(1:nCut)); % check non-conforming edges
    isCheckNode = false(N,1);                
    isCheckNode(cutEdge(checkEdge,1)) = true; % check two end nodes of  
    isCheckNode(cutEdge(checkEdge,2)) = true; % non-conforming edges
    isCheckElem = isCheckNode(elem(1:NT,1)) | isCheckNode(elem(1:NT,2))...
                | isCheckNode(elem(1:NT,3)) | isCheckNode(elem(1:NT,4));
    checkElem = find(isCheckElem); % all elements containing checking nodes
    t2v = sparse(repmat(checkElem,4,1), elem(checkElem,:), 1, NT, N);
    [i,j] = find(t2v(:,cutEdge(checkEdge,1)).*t2v(:,cutEdge(checkEdge,2)));
    markedElem = unique(i);
    nonConforming(checkEdge) = false;
    nonConforming(checkEdge(j)) = true;
    
end
node_out = node(1:N,:); 
elem_out = elem(1:NT,:); 
%% TODO: ifem help doc to explain the bisection algorithm
% <a href="matlab:ifem bisect3">ifem bisect3doc</a>
