# HMM-Using-GMM-
Author: Md Kamrul Hasan
Date: 21th April, 2017


Implement EM to train an HMM (Using GMM data) 

Run instruction: python hmm.py (make sure points.dat in the same directory)



Initialization:
	I have randomly chose k (num of hidden state) data point to initilize k means.
	Intitlize k covariance to make sure determinate is non zero value .
  Initialize transition matrix randomly


Output:
Five files:

  1.training_loglikelihoods.png :
  	 log likelihood on train  vs iteration for different numbers of hidden states. I have used separate means and covariance matrices for each gaussian.
  2.seperate_cov_dev.png:
  	 log likelihood on train  vs iteration for different numbers of mixtures. I have used separate means and covariance matrices for each gaussian.

  3. scatter.png:
     scatter plot for all data

  4. 6 files which plots seperately training log likehood and dev log likelihood for seperate states.



 Result Analysis:

 From scatter plot it can be guessed that the number of cluster should vary among [4,5,6,7]. From log_likelihood graph we can determine the number of appropritae cluster cluster. From both training and dev data loglikehood graph, for which k the graph shows highest log likehood with less fluctuations is good choice for number of clusters. Here, k=5,6 or 7 is almost similar with high log likelihood value. Among them, k=5 converges faster to good log likelihood value with less fluctuations in training set. Morover, in dev set it also converge to high log likehood value within 5 iterations. So, k=5 is good choice.

 HMM model the data better than the original non-sequence model (EM using GMM). It converges faster than the algorithm we applied in previous homework (EM using GMM). It also show less fluctuations in different iterations.   
 
