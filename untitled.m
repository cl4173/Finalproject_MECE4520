T0=273+20;
q=10000/0.005;
K=240;
V=1;
A=0.002;
x=-0.01:0.0001:0.01;
y=-0.01:0.0001:0.01;
R=zeros(201:201);
[X,Y]=meshgrid(x,y);
for i=1:201
    for j=1:201
        R(i,j)=sqrt(x(i)^2+y(j)^2);
    end
end


for t=1:201
    for u=1:201
    T(t,u)=T0+(q/(2*pi*K))*exp(-1*V*x(t)/(2*A))*exp(besselk(0,V*R(t,u)/(2*A)));
    end
end
surf(X,Y,T)