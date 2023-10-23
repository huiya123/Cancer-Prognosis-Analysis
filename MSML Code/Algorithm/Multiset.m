%% Input
% X_all     (n*n_set)×m     Normalized original feature matrices containing all subsets
%           n=n_uncen+n_cen is the number of samples in each subset
%           n_set and m represents the number of subsets and the dimension of features, respectively
% Status    (n*n_set)×1    The event indicator vector
% Time      (n*n_set)×1    The observed time
% Y         (n*n_set)×2    The clinical attribute matrix


%% Parameter Settings
gamma=1e-5;
yita=5*1e-6;
lambda1=1e-4;
lambda2=1e-5;
kappa=1e-4;
epsilon1=1e-9;
epsilon2=1e-9;
epsilon3=1e-5;
epsilon4=1e-2;
epsilon5=1e-3;

theta=1e-3;
nu1=1e-4;
nu2=1e-4;
nu3=1e-2;

c=16;                % The feature dimension for the shared representations in Stage I
e=12;                % The feature dimension for the shared representations in Stage II
n_p=n*(n_set-1);
iter_num=50;         % The number of iterations in Stage I

iter_num_T=50;       % The number of iterations of Teacher model
iter_num_S=50;       % The number of iterations of Student model
iter_num_TS=1000;    % The number of loop iterations of Teacher and Student models

rho=1.2;             % The scale factor 
max_muk=1e6;         % The max value of muk


%% Stage I Multiset representation learning

% Initialization
W=randn(m*n_set,c);    % projection matrix
U=randn(n*n_set,c);    % shared representation matrix
P=randn(n_p*n_set,n);  % reconstruction matrix
J=U;
E1=zeros(n*n_set,c);   % error matrix
E2=zeros(n_p*n_set,c); 
H1=zeros(n*n_set,c);   % multipliers
H2=zeros(n*n_set,c);
H3=zeros(n*n_set,c);

for iter_time=1:iter_num
    %% step1 update U
    XW_1=zeros(n*n_set,c);
    for temp=1:n_set
        XW_1((n*(temp-1)+1):(n*temp),:)=X_all((n*(temp-1)+1):(n*temp),:)*W((m*(temp-1)+1):(m*temp),:);
    end
    U_delta_1=muk*(U+E1-XW_1)-H1;
    
    PPU_2=zeros(n*n_set,c);
    PUN_2=zeros(n*n_set,c);
    PE_2=zeros(n*n_set,c);
    PH_2=zeros(n*n_set,c);
    for temp=1:n_set
        U_new_2=U;
        U_new_2((n*(temp-1)+1):(n*temp),:)=[];
        PPU_2((n*(temp-1)+1):(n*temp),:)=P((n_p*(temp-1)+1):(n_p*temp),:)'*P((n_p*(temp-1)+1):(n_p*temp),:)*U((n*(temp-1)+1):(n*temp),:);
        PUN_2((n*(temp-1)+1):(n*temp),:)=P((n_p*(temp-1)+1):(n_p*temp),:)'*U_new_2;
        PE_2((n*(temp-1)+1):(n*temp),:)=P((n_p*(temp-1)+1):(n_p*temp),:)'*E2((n_p*(temp-1)+1):(n_p*temp),:);
        PH_2((n*(temp-1)+1):(n*temp),:)=P((n_p*(temp-1)+1):(n_p*temp),:)'*H2((n_p*(temp-1)+1):(n_p*temp),:);
    end
    U_delta_2=muk*(PPU_2-PUN_2-PE_2)+PH_2;
    
    U_delta_3=muk*(U-J)+H3;
    
    U_delta_4=zeros(n*n_set,c);
    U_nom=zeros(n*n_set,c);
    for temp=1:n_set 
        U_1=U((n*(temp-1)+1):(n*temp),:);
        U_2=zscore(U_1);
        U_nom((n*(temp-1)+1):(n*temp),:)=U_2;
    end
    for temp=1:n_set
        U_nom_1=U_nom;
        Ui=U_nom((n*(temp-1)+1):(n*temp),:);
        U_nom_1((n*(temp-1)+1):(n*temp),:)=[];
        U_3=zeros(n,c);
        for temp1=1:n_set-1
            Uj=U_nom_1((n*(temp1-1)+1):(n*temp1),:);
            U_4=Uj*Uj'*Ui;
            U_3=U_3+U_4;
        end
        U_delta_4((n*(temp-1)+1):(n*temp),:)=U_3;
    end
    U_delta=U_delta_1+U_delta_2+U_delta_3+kappa*U_delta_4;
    U_new=U-U_delta;
    
    %% step2 update W
    W_new=zeros(m*n_set,c);
    for temp=1:n_set
        W_1=X_all((n*(temp-1)+1):(n*temp),:)'*X_all((n*(temp-1)+1):(n*temp),:)+(2*yita/muk)*eye(m);
        W_2=X_all((n*(temp-1)+1):(n*temp),:)'*U((n*(temp-1)+1):(n*temp),:)+X_all((n*(temp-1)+1):(n*temp),:)'*E1((n*(temp-1)+1):(n*temp),:)-(1/muk)*X_all((n*(temp-1)+1):(n*temp),:)'*H1((n*(temp-1)+1):(n*temp),:);
        W_new((m*(temp-1)+1):(m*temp),:)=W_1\W_2;
    end
    
    %% step3 update P
    P_delta=zeros(n_p*n_set,n);
    for temp=1:n_set
        U_P_update=U;
        U_P_update((n*(temp-1)+1):(n*temp),:)=[];
        P_delta_1=muk*(P((n_p*(temp-1)+1):(n_p*temp),:)*U((n*(temp-1)+1):(n*temp),:)-U_P_update)*U((n*(temp-1)+1):(n*temp),:)';
        P_delta_2=-muk* (E2((n_p*(temp-1)+1):(n_p*temp),:)+(1/muk)*H2((n_p*(temp-1)+1):(n_p*temp),:))*U((n*(temp-1)+1):(n*temp),:)';
        P_delta((n_p*(temp-1)+1):(n_p*temp),:)=P_delta_1+P_delta_2;
    end
    P_new=P-P_delta;
    
    %% step4 update E1
    E1_new=zeros(n*n_set,c);
    for temp=1:n_set
        E1_delta=X_all((n*(temp-1)+1):(n*temp),:)*W((m*(temp-1)+1):(m*temp),:)-U((n*(temp-1)+1):(n*temp),:)+(1/muk)*H1((n*(temp-1)+1):(n*temp),:);
        E1_new((n*(temp-1)+1):(n*temp),:)=sign(E1_delta).*max(abs(E1_delta)-lambda1/muk,0);
    end
    
    %% step5 update E2
    E2_new=zeros(n_p*n_set,c);
    for temp=1:n_set
        U_E2_update=U;
        U_E2_update((n*(temp-1)+1):(n*temp),:)=[];
        E2_delta=P((n_p*(temp-1)+1):(n_p*temp),:)*U((n*(temp-1)+1):(n*temp),:)-U_E2_update+(1/muk)*H2((n_p*(temp-1)+1):(n_p*temp),:);
        E2_new((n_p*(temp-1)+1):(n_p*temp),:)=sign(E2_delta).*max(abs(E2_delta)-lambda2/muk,0);
    end
    
    %% step6 update J
    try
        J_new=zeros(n*n_set,c);
        for temp=1:n_set
            J_delta=U((n*(temp-1)+1):(n*temp),:)+(1/muk)*H3((n*(temp-1)+1):(n*temp),:);
            J_new((n*(temp-1)+1):(n*temp),:)=svso(gamma/muk,J_delta);
        end
    catch
        break;
    end
    
    %% step7 update multipliers
    H1_new=zeros(n*n_set,c);
    H2_new=zeros(n_p*n_set,c);
    H3_new=zeros(n*n_set,c);
    for temp1=1:n_set
        U_H2_update=U;
        U_H2_update((n*(temp2-1)+1):(n*temp2),:)=[];
        H1_1=X_all((n*(temp1-1)+1):(n*temp1),:)*W((m*(temp1-1)+1):(m*temp1),:)-U((n*(temp1-1)+1):(n*temp1),:)-E1((n*(temp1-1)+1):(n*temp1),:);
        H2_1=P((n_p*(temp2-1)+1):(n_p*temp2),:)*U((n*(temp2-1)+1):(n*temp2),:)-U_H2_update-E2((n_p*(temp2-1)+1):(n_p*temp2),:);
        H3_1=U((n*(temp3-1)+1):(n*temp3),:)-J((n*(temp3-1)+1):(n*temp3),:);
        H1_new((n*(temp1-1)+1):(n*temp1),:)=H1((n*(temp1-1)+1):(n*temp1),:)+muk*H1_1;
        H2_new((n_p*(temp2-1)+1):(n_p*temp2),:)=H2_new((n_p*(temp2-1)+1):(n_p*temp2),:)+muk*H2_1;
        H3_new((n*(temp3-1)+1):(n*temp3),:)=H3_new((n*(temp3-1)+1):(n*temp3),:)+muk*H3_1;
    end
    muk=min(rho*muk,max_muk);
    
    
    U=U_new;    J=J_new;    W=W_new;    P=P_new;  E1=E1_new;
    E2=E2_new;  H1=H1_new;  H2=H2_new;  H3=H3_new;
end

%% Mix-supervised multiset fusion model

% initialization
Status_T_1=Status;
Status_T_P=[double(~status_T_1),double(~status_T_1)];
Time_T=Time;

G=randn(c*n_set,e);          
Q= abs(randn(n*n_set,e));
Z=abs(randn(e*n_set,2));
d_T=abs(randn(e*n_set,1));

beta=randn(e,1);           % the coefficient vector for the Cox model 

for TS_num=1:iter_num_TS
    % Teacher
    for iter_time_T1=1:iter_num_T
        %% step1 update Q
        Q_delta=zeros(n*n_set,e);
        for temp=1:n_set
            Q_delta_1=(theta+epsilon1)*Q((n*(temp-1)+1):(n*temp),:)-theta*U((n*(temp-1)+1):(n*temp),:)*G((c*(temp-1)+1):(c*temp),:);
            Q_delta_2=nu1*(Status_T_1((n*(temp-1)+1):(n*temp),:).*(Q((n*(temp-1)+1):(n*temp),:)*d_T((e*(temp-1)+1):(e*temp),:)))*d_T((e*(temp-1)+1):(e*temp),:)';
            Q_delta_3=-nu1*(Status_T_1((n*(temp-1)+1):(n*temp),:) .* Time_train_T((n*(temp-1)+1):(n*temp),:))*d_T((e*(temp-1)+1):(e*temp),:)';
            Q_delta_4=nu2*( Status_T_P((n*(temp-1)+1):(n*temp),:).*(Q((n*(temp-1)+1):(n*temp),:)*Z((e*(temp-1)+1):(e*temp),:)))*Z((e*(temp-1)+1):(e*temp),:)';
            Q_delta_5=-nu2*(Status_T_P((n*(temp-1)+1):(n*temp),:) .* Y((n*(temp-1)+1):(n*temp),:))*Z((e*(temp-1)+1):(e*temp),:)';
            Q_delta((n*(temp-1)+1):(n*temp),:)=Q_delta_1+Q_delta_2+Q_delta_3+Q_delta_4+Q_delta_5;
        end
        Q_new=max(0,Q-Q_delta);
        
        %% step2 update G
        G_delta=zeros(c*n_set,e);
        for temp=1:n_set
            G_delta((c*(temp-1)+1):(c*temp),:)=-theta*U((n*(temp-1)+1):(n*temp),:)'*Q((n*(temp-1)+1):(n*temp),:)+theta*U((n*(temp-1)+1):(n*temp),:)'*U((n*(temp-1)+1):(n*temp),:)* G((c*(temp-1)+1):(c*temp),:)+epsilon2* G((c*(temp-1)+1):(c*temp),:);
        end
        G_new=G-G_delta;
        
        %% step3 update d_T
        d_T_delta=zeros(e*n_set,1);
        for temp=1:n_set
            d_T_delta_1=nu1*Q((n*(temp-1)+1):(n*temp),:)'*(status_T_1((n*(temp-1)+1):(n*temp),:).*(Q((n*(temp-1)+1):(n*temp),:)*d_T((e*(temp-1)+1):(e*temp),:)));
            d_T_delta_2=-nu1*Q((n*(temp-1)+1):(n*temp),:)'*(status_T_1((n*(temp-1)+1):(n*temp),:).*Time_train_T((n*(temp-1)+1):(n*temp),:));
            d_T_delta_3=epsilon3*d_T((e*(temp-1)+1):(e*temp),:);
            d_T_delta((e*(temp-1)+1):(e*temp),:)=d_T_delta_1+d_T_delta_2+d_T_delta_3;
        end
        d_T_new=max(0,d_T-d_T_delta);
        
        %% step4 update Z
        Z_delta=zeros(e*n_set,2);
        for temp=1:n_set
            Z_delta_1=nu2*Q((n*(temp-1)+1):(n*temp),:)'*(Status_T_P((n*(temp-1)+1):(n*temp),:).*(Q((n*(temp-1)+1):(n*temp),:)*Z((e*(temp-1)+1):(e*temp),:)));
            Z_delta_2=-nu2*Q((n*(temp-1)+1):(n*temp),:)'*( Status_T_P((n*(temp-1)+1):(n*temp),:).*Y((n*(temp-1)+1):(n*temp),:));
            Z_delta_3=epsilon4*Z((e*(temp-1)+1):(e*temp),:);
            Z_delta((e*(temp-1)+1):(e*temp),:)=Z_delta_1+Z_delta_2+Z_delta_3;
        end
        Z_new=max(0,Z-Z_delta);
        
        Q=Q_new; G=G_new; Z=Z_new;  d_T=S1_new;
    end
    
    %% Update pseudo labels for censored samples
    Time_S=zeros(n*n_set,1);
    Status_S=zeros(n*n_set,1);
    for temp=1:n_set
        Time_pre1=Q((n*(temp-1)+1):(n*temp),:)*d_T((e*(temp-1)+1):(e*temp),:); % estimation of survival time
        lo_pre_1=find(Status((n*(temp-1)+1):(n*temp),:)==1);
        Time_pre1(lo_pre_1,:)=0;
        
        lo_pre_2=find(Time_pre1>=Time((n*(temp-1)+1):(n*temp),:)); % Locate censored samples whose survival times need to be updated
        Time_S_1=Time((n*(temp-1)+1):(n*temp),:);
        Time_S_1(lo_pre_2,:)=Time_pre1(lo_pre_2,:);
        Time_S((n*(temp-1)+1):(n*temp),:)=Time_S_1;
        
        Status_S_1=Status((n*(temp-1)+1):(n*temp),:);
        Status_S_1(lo_pre_2)=1;
        Status_S((n*(temp-1)+1):(n*temp),:)=Status_S_1;
    end
    
    % Student
    d_S=d_T;
    for iter_time_S=1:iter_num_S
        d_S_delta=zeros(e*n_set,1);
        for temp=1:n_set
            d_S_delta_1=nu3*Q((n*(temp-1)+1):(n*temp),:)'*(results_train_S((n*(temp-1)+1):(n*temp),:).*(Q((n*(temp-1)+1):(n*temp),:)*d_S((e*(temp-1)+1):(e*temp),:)));
            d_S_delta_2=-nu3*Q((n*(temp-1)+1):(n*temp),:)'*(results_train_S((n*(temp-1)+1):(n*temp),:).*Time_train_S((n*(temp-1)+1):(n*temp),:));
            d_S_delta_3=epsilon5*d_S((e*(temp-1)+1):(e*temp),:);
            d_S_delta((e*(temp-1)+1):(e*temp),:)=d_S_delta_1+d_S_delta_2+d_S_delta_3;
        end
        d_S_new=max(0,d_S-d_S_delta);
        d_S=d_S_new;
    end
    
     %% Update pseudo labels for censored samples
    Time_T=zeros(n*n_set,1);
    Status_T=zeros(n*n_set,1);
    for temp=1:n_set
        Time_pre1_S=Q((n*(temp-1)+1):(n*temp),:)*d_S((e*(temp-1)+1):(e*temp),:);
        location_pre_1_S=find(Status((n*(temp-1)+1):(n*temp),:)==1);
        Time_pre1_S(location_pre_1_S,:)=0;
        
        lo_pre_2_S=find(Time_pre1_S>Time((n*(temp-1)+1):(n*temp),:));
        Time_T_1=Time((n*(temp-1)+1):(n*temp),:);
        Time_T_1(lo_pre_2_S,:)=Time_pre1_S(lo_pre_2_S,:);
        Time_T((n*(temp-1)+1):(n*temp),:)=Time_T_1;
        
        Status_T_1=Status((n*(temp-1)+1):(n*temp),:);
        Status_T_1(lo_pre_2_S)=1;
        Status_T((n*(temp-1)+1):(n*temp),:)=Status_T_1;
    end
    
    Status_T_1=Status_T;
    Status_T_P=[double(~Status_T_1),double(~Status_T_1)];
end

% Solve the Cox model
Q_beta_update_1=zeros(n_uncen,e);
Q_beta_update_2=zeros(n_plus*n_set,e);
for temp=1:n_set
    Q_beta_update_1=Q_beta_update_1+Q((n*(temp-1)+1):(n*temp-n_plus),:);
    Q_beta_update_2((n_plus*(temp-1)+1):(n_plus*temp),:)=Q((n*temp-n_plus+1):n*temp ,:);
end
Q_beta_update_1=Q_beta_update_1./n_set;
Q_beta_update=[Q_beta_update_1;Q_beta_update_2]; % feature matrix used to solve the Cox model

Status_cen=zeros(n_plus*n_set,1);
Time_cen=zeros(n_plus*n_set,1);
for temp=1:n_set
    Status_cen((n_plus*(temp-1)+1):(n_plus*temp),:)=results_train_T((n*temp-n_plus+1):n*temp ,:);
    Time_cen((n_plus*(temp-1)+1):(n_plus*temp),:)=Time_train_T((n*temp-n_plus+1):n*temp ,:);
end
Status_beta_update=[ones(n_uncen,1);Status_cen]; % event indicator used to solve the Cox model
Time_beta_update=[Time_train(1:n_uncen,:);Time_cen]; % survival time used to solve the Cox model
data_beta_update=[Q_beta_update,Time_beta_update,Status_beta_update];
data_beta_update=sortrows(data_beta_update,e+1);% Sort samples by survival time

try
    new_beta = solve_beta(data_beta_update(:,e+2),data_beta_update(:,1:e),beta,optim_beta_show);
catch
    disp('error');
end
       
    
    
    
    
    
    