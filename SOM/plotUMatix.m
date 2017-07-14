function [ u_matrix ] = plotUMatix( W, dim, dirName )

%Cria a U-Matriz com zeros
tam_matrix = (2*dim-1);
u_matrix = zeros(tam_matrix);

if any(dim<2)
  return
end

%Define os pontos com as distâncias entre neurônios vizinhos horizontais
for i=1:2:tam_matrix(1)
  for j=2:2:tam_matrix(2)
    indj = j/2;
    indi = floor(i/2+1);
    ind = dim(1)*(indi-1)+indj;
    
    u_matrix(i,j) = norm( W(ind,:) - W(ind+1,:) );
  end
end

%Define os pontos com as distâncias entre neurônios vizinhos verticalmente
for i=2:2:tam_matrix(1)
  for j=1:2:tam_matrix(2)
    indi = i/2;
    indj = floor(j/2+1);
    ind = dim(1)*(indi-1)+indj;
    
    u_matrix(i,j) = norm( W(ind,:) - W(ind+dim(2),:) );
  end
end


%Define os pontos medio das distancias
for i=2:2:tam_matrix
  for j=2:2:tam_matrix
    %média de valores
    valor = u_matrix(i-1,j) + u_matrix(i+1,j) + u_matrix(i,j-1) + u_matrix(i,j+1);
    %u_matrix(i,j) = valor/4;
    
    %como descrito por Costa
    indi = i/2;
    indj = j/2;
    ind = dim(1)*(indi-1)+indj;
    u_matrix(i,j) = 1/2 * 1/sqrt(2) * (  norm(W(ind,:)-W(ind+dim(2)+1,:)) + norm(W(ind+1,:)-W(ind+dim(2),:))  );
    
  end
end




%Define média dos vizinhos de cada neuronio  
for i=1:2:tam_matrix
  for j=1:2:tam_matrix
    
    indi = floor(i/2+1);
    indj = floor(j/2+1);
    ind = dim(1)*(indi-1)+indj;
    
    t = 0;
    soma = 0;
     
    if(indi-1>0) %pra cima
      t = t+1;
      soma = soma + norm(W(ind,:)- W(ind-dim(2),:));
      
      if(indj-1>0) %diagonal cima esquerda
        t = t+1;
        soma = soma + norm(W(ind,:)- W(ind-dim(2)-1,:));
      end
      if(indj+1<=dim(2)) %diagonal cima direita
        t = t+1;
        soma = soma + norm(W(ind,:)- W(ind-dim(2)+1,:));
      end 
    end 
    
    if (indj+1<=dim(2))%pra direita
      t = t+1;
      soma = soma + norm(W(ind,:)- W(ind+1,:));
    end

    if (indi+1<=dim(1))%pra baixo
      t = t+1;
      soma = soma + norm(W(ind,:)- W(ind+dim(2),:));
      
      if(indj-1>0) %diagonal baixo esquerda
        t = t+1;
        soma = soma + norm(W(ind,:)- W(ind+dim(2)-1,:));
      end 
      
      if(indj+1<=dim(2)) %diagonal baixo direita
        t = t+1;
        soma = soma + norm(W(ind,:)- W(ind+dim(2)+1,:));
      end 
    end
    
    if (indj-1>0)%pra esquerda
      t = t+1;
      soma = soma + norm(W(ind,:)- W(ind-1,:));
    end
  
    u_matrix(i,j) = soma/t;
    
  end
end






im_umatrix = figure;
maximo  = max(max(u_matrix));
minimo  = min(min(u_matrix));
data = ((u_matrix-minimo)/(maximo-minimo))*255;
image(data);
colorMap = jet(256);
colormap(colorMap);
colorbar;
title('U-Matrix Bidimensional');

print(im_umatrix,[dirName 'u-matrix2D'],'-dpng');


im_umatrix3D = figure;
surf(1:tam_matrix,1:tam_matrix,u_matrix)
colorbar
title('U-Matrix 3D');

print(im_umatrix3D,[dirName 'u-matrix3D'],'-dpng');





end

