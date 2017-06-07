load data_chainlink
%load data_Test2
%load data_rand1000
%load t4.8k.mat
k=4;

[n,m] = size(X);

X = (X-repmat(mean(X),n,1))./repmat(std(X),n,1); % Z-SCORE NORMALIZATION

%X = X(1:1000,:);

Nx = 12;
Ny = 12;
Ns = Nx*Ny;

[W] = SOM( X, [Nx Ny], 'gauss', .9, 30, 'e', 300 );


%neuronsGrid = [ceil((1:Ns)/dim(1));mod((0:Ns-1),dim(1))+1]';

%plotClusters2D(X,kpoints,U,k,{'X','Y','Z','T'});


grid = (Nx);

for i=1:Ns
  t = 0;
  soma = 0;
  if((fix((i-1)/Nx) ~= 0))%linha pra cima
    t = t+1;
    soma = soma + norm(W(i)-W(i-Nx));
    %plot([W(i,1) W(i-Nx,1)],[W(i,2) W(i-Nx,2)],'r')
  end
  if (rem(i-1,Nx) ~= Nx-1)%linha pra direita
    t = t+1;
    soma = soma + norm(W(i)-W(i+1));
    %plot([W(i,1) W(i+1,1)],[W(i,2) W(i+1,2)],'r')
  end
  if (fix((i-1)/Nx) ~= Nx-1)%linha pra baixo
    t = t+1;
    soma = soma + norm(W(i)-W(i+Nx));
    %plot([W(i,1) W(i+Nx,1)],[W(i,2) W(i+Nx,2)],'r')
  end
  if (rem(i-1,Nx) ~= 0)%linha pra esquerda
    t = t+1;
    soma = soma + norm(W(i)-W(i-1));
    %plot([W(i,1) W(i-1,1)],[W(i,2) W(i-1,2)],'r')
  end
  
  grid((fix((i-1)/Nx)+1), (rem(i-1,Nx)+1)) = soma/t;
end


figure;
maximo  = max(max(grid));
minimo  = min(min(grid));
data = ((grid-minimo)/(maximo-minimo))*255;
image(data);
colorMap = jet(256);
colormap(colorMap);
colorbar;

figure
scatter3(X(:,1), X(:,2), X(:,3))
%scatter(X(:,1),X(:,2))

















%figure
%scatter3(X(:,1), X(:,2), X(:,3))


figure
%[U, kpoints] = kmeans(X,k,'e'); % K-means
scatter3(W(:,1), W(:,2), W(:,3))
%scatter(W(:,1), W(:,2))
hold on
for i=1:Ns
  if((fix((i-1)/Nx) ~= 0))%linha pra cima
    %T(i,i-n,a)
    plot3([W(i,1) W(i-Nx,1)],[W(i,2) W(i-Nx,2)],[W(i,3) W(i-Nx,3)],'r')
    %plot([W(i,1) W(i-Nx,1)],[W(i,2) W(i-Nx,2)],'r')
  end
  if (rem(i-1,Nx) ~= Nx-1)%linha pra direita
    %T(i,i+1,a)
    plot3([W(i,1) W(i+1,1)],[W(i,2) W(i+1,2)],[W(i,3) W(i+1,3)],'r')
    %plot([W(i,1) W(i+1,1)],[W(i,2) W(i+1,2)],'r')
  end
  if (fix((i-1)/Nx) ~= Nx-1)%linha pra baixo
    %T(i,i+n,a)
    plot3([W(i,1) W(i+Nx,1)],[W(i,2) W(i+Nx,2)],[W(i,3) W(i+Nx,3)],'r')
    %plot([W(i,1) W(i+Nx,1)],[W(i,2) W(i+Nx,2)],'r')
  end
  if (rem(i-1,Nx) ~= 0)%linha pra esquerda
    %T(i,i-1,a)
    plot3([W(i,1) W(i-1,1)],[W(i,2) W(i-1,2)],[W(i,3) W(i-1,3)],'r')
    %plot([W(i,1) W(i-1,1)],[W(i,2) W(i-1,2)],'r')
  end
end





%newGrig = zeros(2*Nx-4);
%w2 = reshape(W,[4,4])';


