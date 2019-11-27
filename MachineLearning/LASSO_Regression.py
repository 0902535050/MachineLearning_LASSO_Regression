import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def gradient_descent(x,y):
  lamda=2  
  n_iter=500
  m=len(x)
  l_rate=0.0000001
  er_thres=0.001  
  j_list=[]
  iter_list=[]
  #y_plist=[]  
  theta=[3,3,3,3,3]
  theta=np.array(theta)
  for i in range(n_iter):    
    y_pred=(theta.T.dot(x.T))
    #y_plist.append(y_pred)
    j_theta=(np.sum((y-y_pred)**2))/(2*m)+lamda * np.sum(np.abs(theta))
    j_list.append(j_theta)
    iter_list.append(i)
    grad_theta=(-2/m)*(sum(x.T.dot(y-y_pred)))
    theta=theta-l_rate*grad_theta
    if(j_theta<er_thres and (len(j_list)>10) and np.mean(j_list)):
        break
    elif(j_theta<er_thres):
        break
  return(iter_list,j_list,theta,y_pred)


df=pd.read_csv("train.csv")
df=pd.DataFrame(df)
y=df["PE"] 
x=np.matrix([np.ones(len(y)),df["AT"],df["V"],df["AP"],df["RH"]])
x=pd.DataFrame(x)
x=x.T
a=gradient_descent(x,y)
print("Coefficients are \n theta :{} ".format(a[2]))


print("Plot between no of iteration and error")
plt.plot(a[0],a[1])
plt.show()

plt.plot(df["AT"],np.transpose(a[3]))
plt.show()
#df["AT"]

#source code from https://github.com/sumitsharansatsangi/Lasso_Regression