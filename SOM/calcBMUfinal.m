function [ H, BMUfinal ] = calcBMUfinal( X, W, fDist )

  NX = size(X,1);
  NW = size(W,1);

  H = zeros(NW,1);%histograma de vencedores
  BMUfinal = zeros(NX,1);
  
  for i=1:NX %iteração em todos as instancias
    
    p = X(i,:); % ponto vetorial da instancia i
    Dist = distance(p,W,fDist); % Distancia do ponto p para a posição dos neuronios W
    
    [~,BMU] = min(Dist); %BMU - Neuronio mais próximo de p
    
    H(BMU) = H(BMU)+1;
    BMUfinal(i) = BMU;
  end
end


function D = distance(p, POINTS, fDist)
  [n,~] = size(POINTS);
  P = repmat(p,n,1);
  if strcmp(fDist,'e') %distancia euclidiana
    D = sqrt(sum((P-POINTS).^2, 2));
  elseif strcmp(fDist,'m') %distancia manhattan
    D = sum(abs(P-POINTS), 2);
  end
end
