function [ ] = plotClusters4D( data,kpoints,U,k,labels )

% SEPARA OS INDICES DOS DADOS DE CADA CLUSTER
clusters = cell(k,1);
for i=1:k
  clusters{i} = find(U(i,:)==1);
end

markers = {'+','o','*','x','s','d','^','v','>','<','p','h'};

% PLOTA GRÁFICOS 3D
figure
igraf = 0;
for x=1:2
for y=2:3
for z=3:4
  if (x~=y && x~=z && y~=z)
    igraf=igraf+1;
    subplot(2,2,igraf)
    for i=1:k %polota dados de cada cluster
      scatter3(data(clusters{i},x), data(clusters{i},y),data(clusters{i},z), markers{mod(i,numel(markers))+1})
      hold on
    end
    scatter3(kpoints(:,x),kpoints(:,y),kpoints(:,z),40,'k','filled') %plota centroids
    
    if(iscellstr(labels)) %escreve labels
      xlabel(labels(x));
      ylabel(labels(y));
      zlabel(labels(z));
    end
  end
end
end
end



% PLOTA GRÁFICOS 2D
figure
igraf = 0;
for y=1:4
for x=1:4
  igraf=igraf+1;
  if (x~=y)
    subplot(4,4,igraf)
    for i=1:k %polota dados de cada cluster
      scatter(data(clusters{i},x), data(clusters{i},y), markers{mod(i,numel(markers))+1})
      hold on
    end
    scatter(kpoints(:,x),kpoints(:,y),40,'k','filled') %plota centroids
  else
    if(iscellstr(labels)) %escreve labels
      subplot(4,4,igraf)
      text(.5,.5,labels(y),'FontSize',12);
      set(gca,'visible','off');
    end
  end
end
end


end

