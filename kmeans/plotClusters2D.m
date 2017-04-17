function [ ] = plotClusters2D( data,kpoints,U,k,labels )

clusters = cell(k,1);
for i=1:k % Separa os indices dos dados de cada cluster
  clusters{i} = find(U(i,:)==1);
end

figure
hold all
markers = {'+','o','*','x','s','d','^','v','>','<','p','h'};

for i=1:k %polota dados de cada cluster
  scatter(data(clusters{i},1), data(clusters{i},2), markers{mod(i,numel(markers))+1})
  %fprintf('Cluster %i = %s\n', i, markers{mod(i,numel(markers))+1});
end

scatter(kpoints(:,1),kpoints(:,2),40,'k','filled') %plota centroids

if(iscellstr(labels)) %escreve labels
  xlabel(labels(1));
  ylabel(labels(2));
end

end

