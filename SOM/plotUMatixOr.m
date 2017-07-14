function [ u_matrix ] = plotUMatix( W, Nx, dirName )

%Cria a U-Matriz com zeros
tam_matrix = (2*Nx-1);
u_matrix = zeros(tam_matrix);


%Define os pontos com as distâncias entre neurônios vizinhos horizontais
for i=1:2:tam_matrix
  for j=2:2:tam_matrix
    indj = j/2;
    indi = floor(i/2+1);
    ind = Nx*(indi-1)+indj;
    
    u_matrix(i,j) = norm( W(ind) - W(ind+1) );
  end
end

%Define os pontos com as distâncias entre neurônios vizinhos verticalmente
for i=2:2:tam_matrix
  for j=1:2:tam_matrix
    indi = i/2;
    indj = floor(j/2+1);
    ind = Nx*(indi-1)+indj;
    
    u_matrix(i,j) = norm( W(ind) - W(ind+1) );
  end
end

%Define os pontos medio das distancias
for i=2:2:tam_matrix
  for j=2:2:tam_matrix
    valor = u_matrix(i-1,j) + u_matrix(i+1,j) + u_matrix(i,j-1) + u_matrix(i,j+1);
    
    u_matrix(i,j) = valor/4;
  end
end


%Define média dos vizinhos de cada neuronio  
for i=1:2:tam_matrix
  for j=1:2:tam_matrix
    t = 0;
    soma = 0;
    if(i-1>0) %pra cima
      t = t+1;
      soma = soma + u_matrix(i-1,j);
    end 
    if (j+1<=Nx)%pra direita
      t = t+1;
      soma = soma + u_matrix(i,j+1);
    end
    if (i+1<=Nx)%pra baixo
      t = t+1;
      soma = soma + u_matrix(i+1,j);
    end
    if (j-1>0)%pra esquerda
      t = t+1;
      soma = soma + u_matrix(i,j-1);
    end
  
    u_matrix(i,j) = soma/t;
    
  end
end






im_grid = figure;
maximo  = max(max(u_matrix));
minimo  = min(min(u_matrix));
data = ((u_matrix-minimo)/(maximo-minimo))*255;
image(data);
colorMap = jet(256);
colormap(colorMap);
colorbar;

print(im_grid,[dirName 'u-matrix'],'-dpng');





return






%Define média dos vizinhos de cada neuronio  
for i=1:2:tam_matrix
  for j=1:2:tam_matrix
    t = 0;
    soma = 0;
    if((fix((i-1)/Nx) ~= 0))%linha pra cima
      t = t+1;
      soma = soma + norm(W(i)-W(i-Nx));
    end 
    if (rem(i-1,Nx) ~= Nx-1)%linha pra direita
      t = t+1;
      soma = soma + norm(W(i)-W(i+1));
    end
    if (fix((i-1)/Nx) ~= Nx-1)%linha pra baixo
      t = t+1;
      soma = soma + norm(W(i)-W(i+Nx));
    end
    if (rem(i-1,Nx) ~= 0)%linha pra esquerda
      t = t+1;
      soma = soma + norm(W(i)-W(i-1));
    end
  
    grid((fix((i-1)/Nx)+1), (rem(i-1,Nx)+1)) = soma/t;
    
  end
end





%Define valores faltantes com a média das distancias



for i=1:(2*Nx-1)
  for j=1:(2*Nx-1)
    
    if mod(i,2)~=0 %linha impar
      if mod(j,2)~=0 %coluna impar
        
        
        
        
        
      else %coluna par
        
      end
      
    else %linha par
      if mod(j,2)~=0 %coluna impar
        
      else %coluna par
      end
    end
      
  end
end






grid = (Nx);

for i=1:Ns
  t = 0;
  soma = 0;
  if((fix((i-1)/Nx) ~= 0))%linha pra cima
    t = t+1;
    soma = soma + norm(W(i)-W(i-Nx));
  end 
  if (rem(i-1,Nx) ~= Nx-1)%linha pra direita
    t = t+1;
    soma = soma + norm(W(i)-W(i+1));
  end
  if (fix((i-1)/Nx) ~= Nx-1)%linha pra baixo
    t = t+1;
    soma = soma + norm(W(i)-W(i+Nx));
  end
  if (rem(i-1,Nx) ~= 0)%linha pra esquerda
    t = t+1;
    soma = soma + norm(W(i)-W(i-1));
  end

  grid((fix((i-1)/Nx)+1), (rem(i-1,Nx)+1)) = soma/t;
end



im_grid = figure;
maximo  = max(max(grid));
minimo  = min(min(grid));
data = ((grid-minimo)/(maximo-minimo))*255;
image(data);
colorMap = jet(256);
colormap(colorMap);
colorbar;

print(im_grid,[dirName 'u-matrix'],'-dpng');

end

