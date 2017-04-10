function [ ] = plotClustersIris( data,kpoints,k1,k2,k3 )

labels = {'sepal length' 'sepal width' 'petal length' 'petal width'};

figure

% SEPAL LENGTH LINE
subplot(4,4,1)
text(.25,.5,'SEPAL LENGTH','FontSize',12);
set(gca,'visible','off');

subplot(4,4,2)
plot(data(k1,2), data(k1,1), 'ro')
hold on
plot(data(k2,2), data(k2,1), 'bx')
plot(data(k3,2), data(k3,1), 'gd')
scatter(kpoints(:,2),kpoints(:,1),40,'k','filled')
%xlabel(labels(2));
%ylabel(labels(1));
%set(gca,'XtickLabel',[],'YtickLabel',[]);

subplot(4,4,3)
plot(data(k1,3), data(k1,1), 'ro')
hold on
plot(data(k2,3), data(k2,1), 'bx')
plot(data(k3,3), data(k3,1), 'gd')
scatter(kpoints(:,3),kpoints(:,1),40,'k','filled')
%xlabel(labels(3));
%ylabel(labels(1));
%set(gca,'XtickLabel',[],'YtickLabel',[]);

subplot(4,4,4)
plot(data(k1,4), data(k1,1), 'ro')
hold on
plot(data(k2,4), data(k2,1), 'bx')
plot(data(k3,4), data(k3,1), 'gd')
scatter(kpoints(:,4),kpoints(:,1),40,'k','filled')
%xlabel(labels(4));
%ylabel(labels(1));
%set(gca,'XtickLabel',[],'YtickLabel',[]);

% SEPAL WIDTH LINE
subplot(4,4,5)
plot(data(k1,1), data(k1,2), 'ro')
hold on
plot(data(k2,1), data(k2,2), 'bx')
plot(data(k3,1), data(k3,2), 'gd')
scatter(kpoints(:,1),kpoints(:,2),40,'k','filled')
%xlabel(labels(1));
%ylabel(labels(2));
%set(gca,'XtickLabel',[],'YtickLabel',[]);

subplot(4,4,6)
text(.25,.5,'SEPAL WIDTH','FontSize',12);
set(gca,'visible','off');

subplot(4,4,7)
plot(data(k1,3), data(k1,2), 'ro')
hold on
plot(data(k2,3), data(k2,2), 'bx')
plot(data(k3,3), data(k3,2), 'gd')
scatter(kpoints(:,3),kpoints(:,2),40,'k','filled')
%xlabel(labels(3));
%ylabel(labels(2));
%set(gca,'XtickLabel',[],'YtickLabel',[]);

subplot(4,4,8)
plot(data(k1,4), data(k1,2), 'ro')
hold on
plot(data(k2,4), data(k2,2), 'bx')
plot(data(k3,4), data(k3,2), 'gd')
scatter(kpoints(:,4),kpoints(:,2),40,'k','filled')
%xlabel(labels(4));
%ylabel(labels(2));
%set(gca,'XtickLabel',[],'YtickLabel',[]);

% PETAL LENGTH LINE

subplot(4,4,9)
plot(data(k1,1), data(k1,3), 'ro')
hold on
plot(data(k2,1), data(k2,3), 'bx')
plot(data(k3,1), data(k3,3), 'gd')
scatter(kpoints(:,1),kpoints(:,3),40,'k','filled')
%xlabel(labels(1));
%ylabel(labels(3));
%set(gca,'XtickLabel',[],'YtickLabel',[]);

subplot(4,4,10)
plot(data(k1,2), data(k1,3), 'ro')
hold on
plot(data(k2,2), data(k2,3), 'bx')
plot(data(k3,2), data(k3,3), 'gd')
scatter(kpoints(:,2),kpoints(:,3),40,'k','filled')
%xlabel(labels(2));
%ylabel(labels(3));
%set(gca,'XtickLabel',[],'YtickLabel',[]);

subplot(4,4,11)
text(.25,.5,'PETAL LENGTH','FontSize',12);
set(gca,'visible','off');

subplot(4,4,12)
plot(data(k1,4), data(k1,3), 'ro')
hold on
plot(data(k2,4), data(k2,3), 'bx')
plot(data(k3,4), data(k3,3), 'gd')
scatter(kpoints(:,4),kpoints(:,3),40,'k','filled')
%xlabel(labels(4));
%ylabel(labels(3));
%set(gca,'XtickLabel',[],'YtickLabel',[]);

% PETAL WIDTH LINE

subplot(4,4,13)
plot(data(k1,1), data(k1,4), 'ro')
hold on
plot(data(k2,1), data(k2,4), 'bx')
plot(data(k3,1), data(k3,4), 'gd')
scatter(kpoints(:,1),kpoints(:,4),40,'k','filled')
%xlabel(labels(1));
%ylabel(labels(4));
%set(gca,'XtickLabel',[],'YtickLabel',[]);

subplot(4,4,14)
plot(data(k1,2), data(k1,4), 'ro')
hold on
plot(data(k2,2), data(k2,4), 'bx')
plot(data(k3,2), data(k3,4), 'gd')
scatter(kpoints(:,2),kpoints(:,4),40,'k','filled')
%xlabel(labels(2));
%ylabel(labels(4));
%set(gca,'XtickLabel',[],'YtickLabel',[]);

subplot(4,4,15)
plot(data(k1,3), data(k1,4), 'ro')
hold on
plot(data(k2,3), data(k2,4), 'bx')
plot(data(k3,3), data(k3,4), 'gd')
scatter(kpoints(:,3),kpoints(:,4),40,'k','filled')
%xlabel(labels(3));
%ylabel(labels(4));
%set(gca,'XtickLabel',[],'YtickLabel',[]);

subplot(4,4,16)
text(.25,.5,'PETAL WIDTH','FontSize',12);
set(gca,'visible','off');

end

