import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def cost(X,theta,y):
	m=np.size(X)/2
	X=X.dot(theta)
	#print("dot",X)
	htheta2=(X-y)**2
	#print("square",htheta2)
	cc=np.sum(htheta2)
	return cc/(2*m)

data=pd.read_csv('data.csv')

x=np.array(data.X,'float64')
y=np.array(data.Y,'float64')

plt.scatter(x,y)

theta=np.array([0.0,0.0]);
alpha=0.000003
m=np.size(x)
X=np.array([np.ones(m),x]).T
cst=[]
for i in range(100):
	theta0=theta[0]-(alpha/m)*(np.sum(X.dot(theta)-y))
	theta1=theta[1]-(alpha/m)*(np.sum((X.dot(theta)-y)*x))
	theta[0]=theta0
	theta[1]=theta1
	cst.append(cost(X,theta,y)/10)

y_predicted=X.dot(theta)	
print(f"Parameters: Theta0={theta[0]} ,Theta1={theta[1]}")
print("Predicted y",y_predicted)
print(f"cost after {i+1} iteration is {cost(X,theta,y)}")
#plt.plot(range(70),cst,'g')
plt.plot(x,y_predicted,'r')
plt.show()
