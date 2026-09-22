
%Defining initial parameters
T = 20;
flasht = 0;
half_width = 0.0002;
rho = 4500;
cut_force = 50;
tool_width = 0.003;
cut_speed = 1;
iteration = 0;
tplot = zeros(1000,1);

while true %Iteration to find flash temp
        iteration = iteration +1;
        specific_heat = ((1.71*10^-6)*(T^3))-(0.00173*(T^2))+(0.6*T)+536;
        thermal_cond = ((1.12*10^-5)*(T^2))+(0.00982*T)+6.4;
        Pe = (cut_speed*rho*half_width*specific_heat)/(2*thermal_cond);
        c4 = (0.00527*(Pe^3))-(0.192*(Pe^2))+2.39*Pe;
        if Pe<5
            flasht = round((0.159*c4)*((2*cut_force)/(rho*specific_heat*tool_width*half_width)));
        else
            flasht = round((0.399)*((2*cut_force*cut_speed)/(thermal_cond*tool_width))*(sqrt((thermal_cond)/(rho*specific_heat*cut_speed*half_width))));
        end
        if flasht == T
            break;
        end
        tplot(iteration,1) = flasht;
        T=(flasht+T)/2;
end
tplot = tplot(1:iteration-1,1);
Pep = ['Peclet number (Pe) =' ,num2str(Pe)];
disp(Pep); 
flashtp = ['Flash temperature =',num2str(flasht)];
disp(flashtp);
plot(tplot);                                                                    %Convergence plot for flash temp
figure;

%Finding temp profile on boundary
x_locations = zeros(41,1);
xoverb_start = -2.001;
gap = 0.1;
flank = (1 - xoverb_start)/gap;
p_plus = zeros(length(x_locations),1);
p_minus = zeros(length(x_locations),1);
sst = zeros(length(x_locations),1);
for i = 1:length(x_locations)
    x_locations(i) = xoverb_start;
    p_plus(i) = (Pe*(x_locations(i)+1))/2;
    p_minus(i) = (Pe*(x_locations(i)-1))/2;
    if x_locations(i)<-1
        sst(i)=real((x_locations(i)+1)*exp(p_plus(i))*(besselk(0,-p_plus(i))-besselk(1,-p_plus(i)))+(1-x_locations(i))*exp(p_minus(i))*(besselk(0,-p_minus(i))-besselk(1,-p_minus(i))));
    elseif (-1 < x_locations(i))<1
        sst(i)=real(((x_locations(i)+1)*exp(p_plus(i))*(besselk(0,p_plus(i))+besselk(1,p_plus(i))))+((1-x_locations(i))*exp(p_minus(i))*(besselk(0,-p_minus(i))-besselk(1,-p_minus(i)))));
    else
        sst(i)=real(((x_locations(i)+1)*exp(p_plus(i))*(besselk(0,p_plus(i))+besselk(1,p_plus(i))))+((1-x_locations(i))*exp(p_minus(i))*(besselk(0,p_minus(i))+besselk(1,p_minus(i)))));
    end
    xoverb_start = xoverb_start + gap;
end
%Finding the temp distribution in the subsurface
gap_z = 0.01;
z_locations = zeros(400,1);
k=0;
for j = 1:400
    k=k+gap_z;
    z_locations(j) = k;
end
maxt=max(sst);
subsurface_temp = zeros(length(sst),1);
normalized_sst = zeros(length(sst),1);
for l = 1:length(sst)
    normalized_sst(l) = sst(l)/maxt;
    subsurface_temp(l) = flasht*normalized_sst(l)+20;
end  
plot(x_locations,subsurface_temp);                  %temp profile plot
figure;
all_temp = zeros(length(sst),length(z_locations));
for m = 1:length(sst)
    for n = 1:length(z_locations)
        if (1-erf((z_locations(n)*half_width)/sqrt((8*thermal_cond*half_width)/(specific_heat*rho*cut_speed))))*subsurface_temp(m)<20
            all_temp(m,n)=20;
        else
            all_temp(m,n)=(1-erf((z_locations(n)*half_width)/sqrt((8*thermal_cond*half_width)/(specific_heat*rho*cut_speed))))*subsurface_temp(m);
        end
    end
end       
plot(all_temp);                                                                 %subsurface plot
figure;
Tflank_temp = all_temp(round(flank)+1,:);
Tflank_temp = Tflank_temp';
%disp(Tflank_temp);
%finding residual stresses
residual = zeros(1000,7);
temp=20; youngs=0; ys=0; expansivity=0; estress=0; v=0.32;
for p = 1:length(residual)
    residual(p,1) = temp;
    youngs = (-4*10*temp+1*100000);
    residual(p,2) = youngs;
    expansivity = (1.23*10^-6*temp^2+2.87*10^-3*temp+4.57)*1.8*10^-6;
    residual(p,3) = expansivity;
    ys = (0.005752*temp^4-10.67*temp^3+5597*temp^2-1.79*10^6*temp+1.056*10^9)*10^-6;
    residual(p,4) = ys;
    estress=youngs*expansivity*(temp-20)/(1-v);
    residual(p,5) = estress;
    if estress>ys
        residual(p,6) = estress - ys;
        residual(p,7) = temp;
    else
        residual(p,6) =0;
    end
    temp = temp +1;
end
tempcol=residual(:,7);
toplot = residual(:,[4,5,6]);
Pep = ['Peclet number (Pe) =' ,num2str(Pe)];
Tcritical = min(tempcol(tempcol>0));
Tcriticalp = ['Critical temperature = ',num2str(Tcritical)];
%disp(residual);
disp(Tcriticalp);
plot(toplot);                                                              %Residual stress Plot at surface
%Finding residual stress at tool exit
residual_stress = zeros(length(Tflank_temp),3);
for j = 1:length(Tflank_temp)
    residual_stress(j,2) = Tcritical;
    residual_stress(j,3) = Tflank_temp(j);
    if Tflank_temp(j)> Tcritical
        residual_stress(j,1) = 2.8788*Tflank_temp(j)-1365.2;
    else
        residual_stress(j,1) =0;
    end
end
plot(residual_stress);                                                      %Residual stress at subsurfaces












