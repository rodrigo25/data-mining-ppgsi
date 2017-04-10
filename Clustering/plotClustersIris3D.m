function [ ] = plotClustersIris3D( data,kpoints,k1,k2,k3 )

labels = {'sepal length' 'sepal width' 'petal length' 'petal width'};

figure

subplot(2,2,1)
scatter3(data(k1,1), data(k1,2), data(k1,3), 'ro')
hold on
scatter3(data(k2,1), data(k2,2), data(k2,3), 'bx')
scatter3(data(k3,1), data(k3,2), data(k3,3), 'gd')
scatter3(kpoints(:,1),kpoints(:,2),kpoints(:,3),40,'k','filled')
xlabel(labels(1));
ylabel(labels(2));
zlabel(labels(3));


subplot(2,2,2)
scatter3(data(k1,1), data(k1,2), data(k1,4), 'ro')
hold on
scatter3(data(k2,1), data(k2,2), data(k2,4), 'bx')
scatter3(data(k3,1), data(k3,2), data(k3,4), 'gd')
scatter3(kpoints(:,1),kpoints(:,2),kpoints(:,4),40,'k','filled')
xlabel(labels(1));
ylabel(labels(2));
zlabel(labels(4));


subplot(2,2,3)
scatter3(data(k1,1), data(k1,3), data(k1,4), 'ro')
hold on
scatter3(data(k2,1), data(k2,3), data(k2,4), 'bx')
scatter3(data(k3,1), data(k3,3), data(k3,4), 'gd')
scatter3(kpoints(:,1),kpoints(:,3),kpoints(:,4),40,'k','filled')
xlabel(labels(1));
ylabel(labels(3));
zlabel(labels(4));


subplot(2,2,4)
scatter3(data(k1,2), data(k1,3), data(k1,4), 'ro')
hold on
scatter3(data(k2,2), data(k2,3), data(k2,4), 'bx')
scatter3(data(k3,2), data(k3,3), data(k3,4), 'gd')
scatter3(kpoints(:,2),kpoints(:,3),kpoints(:,4),40,'k','filled')
xlabel(labels(2));
ylabel(labels(3));
zlabel(labels(4));

end

