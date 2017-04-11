function [ ] = plotClusters3D( data,kpoints,U,k,labels )

clusters = cell(k,1);
for i=1:k % Separa os indices dos dados de cada cluster
  clusters{i} = find(U(i,:)==1);
end

figure
hold all
markers = {'+','o','*','x','s','d','^','v','>','<','p','h'};

for i=1:k %polota dados de cada cluster
  scatter3(data(clusters{i},1), data(clusters{i},2),data(clusters{i},3), markers{mod(i,numel(markers))+1})
end

scatter3(kpoints(:,1),kpoints(:,2),kpoints(:,3),40,'k','filled') %plota centroids

if(iscellstr(labels)) %escreve labels
  xlabel(labels(1));
  ylabel(labels(2));
  zlabel(labels(3));
end

end

