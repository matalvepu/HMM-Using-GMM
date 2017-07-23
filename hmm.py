import numpy as np 
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
import random

#initialize means,covariences and transition matrix 
def init_hmm_params(data,num_hidden_states):
	num_dim=len(data[0])
	num_data=len(data)
	means=[]
	#randomly pick point from data set
	for i in range(num_hidden_states):
		index=random.randint(0,len(data)-1)
		means.append(data[index])
	means=np.array(means)

	#randomly generated number
	covs=[]
	for k in range(num_hidden_states):
		cov=np.zeros(shape=(num_dim,num_dim))
		for j in range(num_dim):
			cov[j][j]=float(random.randint(1,80))/80
		covs.append(cov.tolist())

	covs=np.array(covs)

	#transition matrix
	a=[]
	for i in range(num_hidden_states):
		t_prob=np.random.random(num_hidden_states)
		t_prob /= t_prob.sum()
		a.append(t_prob.tolist())
	a=np.array(a)	
	return means,covs,a

#forward algorithm to calculate alpha
def forward(data,a,means,covs):
	num_state=len(means)
	num_data=len(data)
	alpha=np.zeros(shape=(num_state,num_data),dtype=np.float128)
	#initialization for first observation
	for s in range(num_state):
		p=multivariate_normal.pdf([data[0]], mean=means[s],cov=covs[s]);
		alpha[s][0]=(float(1)/num_state) * p
	#recursively populate other alpha
	for t in range(1,num_data):
		for s in range(num_state):
			obs_prob_t=multivariate_normal.pdf([data[t]], mean=means[s],cov=covs[s]);
			alpha[s][t]=0
			for s1 in range(num_state):
				alpha[s][t]+=alpha[s1][t-1]*a[s1][s]*obs_prob_t

	return alpha

#compute beta
def backword(data,a,means,covs):
	num_state=len(means)
	num_data=len(data)

	#initialization for last observation
	beta=np.zeros(shape=(num_state,num_data),dtype=np.float128)
	for s in range(num_state):
		beta[s][num_data-1]=1.

	for t in range(num_data-2,-1,-1):
		for s in range(num_state):		
			beta[s][t]=0.
			for s1 in range(num_state):
				obs_prob_t=multivariate_normal.pdf([data[t+1]], mean=means[s1],cov=covs[s1]);
				beta[s][t]+=beta[s1][t+1]*obs_prob_t*a[s][s1]

	return beta
#compute gamma=num_stae*num_data
#gamma(s,t)=it is the probability of observating t in sate s
def compute_gamma(data,alpha,beta):
	num_state=len(means)
	num_data=len(data)

	gamma=np.zeros(shape=(num_state,num_data),dtype=np.float128)

	for t in range(num_data):
		s=0.
		for j in range(num_state):
			val=alpha[j][t]*beta[j][t]
			gamma[j][t]=val
			s+=val

		for j in range(num_state):
			if s!=0:
				gamma[j][t]/=s

			
	return gamma

#Compute eta=num_state*num_state*num_data
# eta(i,j,t)= probability of being in state i at time t and being in state j in t+1
def compute_eta(data,a,alpha,beta,means,covs):
	num_state=len(means)
	num_data=len(data)
	eta=np.zeros(shape=(num_state,num_state,num_data),dtype=np.float128)

	for t in range(num_data-1):
		s=0.
		for i in range(num_state):
			for j in range(num_state):
				p=multivariate_normal.pdf([data[t+1]], mean=means[j],cov=covs[j]);
				val=alpha[i][t]*a[i][j]*p*beta[j][t+1]
				eta[i][j][t]=val
				s+=val

		for i in range(num_state):
			for j in range(num_state):
				eta[i][j][t]/=s

	return eta

#update transition matrix using eta
def update_transition_matrix(num_state,num_data,eta):

	a=np.zeros(shape=(num_state,num_state),dtype=np.float128)
	for i in range(num_state):
		s=0.		
		for j in range(num_state):
			val=0.
			for t in range(num_data-1):
				val+=eta[i][j][t]

			a[i][j]=val
			s+=val

		for j in range(num_state):
			if s!=0:
				a[i][j]/=s
	return a

def update_means(data,num_state,gamma):
	num_dim=len(data[0])
	num_data=len(data)
	means=[]
	
	for j in range(num_state):
		n=np.zeros(shape=(num_dim),dtype=np.float128)
		for t in range(num_data):
			n+=gamma[j][t]*data[t]

		d=0.
		for t in range(num_data):
			d+=gamma[j][t]

		if d!=0:
			n/=d
		means.append(n)

	return np.array(means)
	

def update_covs(data,num_state,gamma,means):
	num_data=len(data)
	num_dim=len(data[0])
	covs= [np.zeros((num_dim,num_dim))] * num_state

	for j in range(num_state):
		sum_j=np.zeros((num_dim,num_dim))
		for t in range(num_data):
			x=data[t]-means[j]
			x=gamma[j][t]*np.outer(x,x)
			sum_j+=x

		d=0.
		for t in range(num_data):
			d+=gamma[j][t]

		covs[j]=sum_j/d

	return np.array(covs)




def m_step(gamma,eta,means,covs,data):
	num_state=len(means)
	num_data=len(data)

	a=update_transition_matrix(num_state,num_data,eta)
	means=update_means(data,num_state,gamma)
	covs=update_covs(data,num_state,gamma,means)

	return a,means,covs



def e_step(data,a,means,covs):

	alpha=forward(data,a,means,covs)
	beta=backword(data,a,means,covs)

	gamma=compute_gamma(data,alpha,beta)

	eta=compute_eta(data,a,alpha,beta,means,covs)

	return gamma,eta


def log_likelihood(data,means,covs,mixing_coeffs):
	ll=0
	num_clusters = len(means)
	for t in range(len(data)):
		sum_resp=0
		for j in range(num_clusters):
			p=multivariate_normal.pdf([data[t]], mean=means[j],cov=covs[j]);
			sum_resp+=mixing_coeffs[j]*p
		ll+=np.log(sum_resp)
	return ll

def compute_mixing_coeffs(gamma,data):
	num_data=len(data)
	num_state=len(gamma)

	pi_k=np.zeros(shape=(num_state),dtype=np.float128)

	for k in range(num_state):
		s=0.
		for i in range(num_data):
			s+=gamma[k][i]

		pi_k[k]=s

	for k in range(num_state):
		pi_k[k]/=num_data
		
	return pi_k



if __name__ == "__main__":

	data=np.loadtxt("points.dat")
	training_data=data[0:899]
	dev_data=data[900:999]
	maxiter=40
	K=list(range(2,8))

	all_t_ll=[]
	all_d_ll=[]
	#run for different number of hidden states
	for num_hidden_states in K:
		
		means,covs,a=init_hmm_params(training_data,num_hidden_states)
		training_ll=[]
		dev_ll=[]

		for itr in range(maxiter):
			#e step
			gamma,eta=e_step(training_data,a,means,covs)

			#m step
			a,means,covs=m_step(gamma,eta,means,covs,training_data)
			
			mixing_coeffs=compute_mixing_coeffs(gamma,training_data)

			t_ll=log_likelihood(training_data,means,covs,mixing_coeffs)
			training_ll.append(t_ll)
			
			d_ll=log_likelihood(dev_data,means,covs,mixing_coeffs)
			dev_ll.append(d_ll)


		all_t_ll.append(training_ll)
		all_d_ll.append(dev_ll)

	#plot all log likehoods for different num of hidden states together
	legend=[]
	for i in range(len(all_t_ll)):
		plt.plot(all_t_ll[i],linestyle='--', marker='.')
		legend.append('#hidden state='+str(i+2))

	plt.ylabel('log likelihood in training data (diff num of hidden states)')
	plt.xlabel('iteration')
	plt.legend(legend, loc='lower right')
	plt.show()

	legend=[]
	for i in range(len(all_d_ll)):
		plt.plot(all_d_ll[i],linestyle='--', marker='.')
		legend.append('#hidden state='+str(i+2))

	plt.ylabel('log likelihood in dev data (diff num of hidden states)')
	plt.xlabel('iteration')
	plt.legend(legend, loc='lower right')
	plt.show()
	#plot log likehoods for different num of hidden states seperately
	for i in range(len(K)):

		plt.subplot(2, 1, 1)
		plt.title('# hidden states='+str(i+2))
		plt.plot(all_t_ll[i],linestyle='--', marker='.',c='b')
		plt.ylabel('log likelihood in training set')

		plt.subplot(2, 1, 2)
		plt.plot(all_d_ll[i],linestyle='--', marker='.',c='r')
		plt.ylabel('log likelihood in dev set')
		plt.xlabel('iteration')
		plt.show()



