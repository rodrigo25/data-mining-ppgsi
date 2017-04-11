function [ U, medoidPoints ] = PAM( data, k, dist )

  [n, m] = size(data);
  
  Distancias = zeros(n,2);
  kmedoide = zeros(k,1);
  
  %Cria matrix de distancias
  for i=1:n   %para cada instancia
    point = data(i,:);
    D(:,i) = distance(point, data, n, dist); %calcula distancias do ponto a todos os outros
  end
  
  %Seleciona medoides iniciais
  [~,medoide] = min(sum(D)); %primeiro medoide
  kmedoide(1) = medoide; 
  Distancias(:,1) = D(:,medoide); %distancias dos dados ao primeiro medoide
  Distancias(:,2) = medoide;
  Distancias(medoide,2) = 0;
  
  for i=2:k
    [~,medoide] = max(Distancias(:,1));
    kmedoide(i) = medoide;
    Distancias(D(:,medoide)<Distancias(:,1),2) = medoide;
    Distancias(D(:,medoide)<Distancias(:,1),1) = D(D(:,medoide)<Distancias(:,1),medoide);
    Distancias(medoide,2) = 0;
  end
  
  
  %SWAP
  while(true)
    for i=1:k
      medoide = kmedoide(i);
      cost_i = sum(Distancias(1,Distancias(:,2)==medoide));
      for j=1:n
        
      end
    end
    costMedoids_old = costMedoids;
    costMedoids = zeros(k,1);
    
    
  end
  
end


% FUNÇÃO DE DISTÂNCIAS
function D = distance(p, data, n, dist)
  P = repmat(p,n,1);
  if strcmp(dist,'e') %distancia euclidiana
    D = sqrt(sum((P-data).^2, 2));
  elseif strcmp(dist,'m') %distancia manhattan
    D = sum(abs(P-data), 2);
  end
end

