load fisheriris.mat
k=3;

[n,m] = size(meas);
%meas_normaliz = (meas-repmat(mean(meas),n,1))./repmat(std(meas),n,1);

min_val = min(meas);
max_val = max(meas);
for i=1:m
  meas(:,i) = (meas(:,i)-min(i))/(max(i)-min(i));
end

[U, kpoints] = kmeans(meas_normaliz,k);

fprintf('Quantidade de Exemplos por cluster\n');
fprintf('Exemplos do cluster 1: %i\n',length(find(U(1,:)==1)));
fprintf('Exemplos do cluster 2: %i\n',length(find(U(2,:)==1)));
fprintf('Exemplos do cluster 3: %i\n',length(find(U(3,:)==1)));

fprintf('\n\n\nClasses dos exemplos do cluster 1:\n')
pause;
species(find(U(1,:)==1))

fprintf('\n\n\nClasses dos exemplos do cluster 2:\n')
pause;
species(find(U(2,:)==1))

fprintf('\n\n\nClasses dos exemplos do cluster 3:\n')
pause;
species(find(U(3,:)==1))