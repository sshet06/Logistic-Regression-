# Logistic Regression

### Input:
This program implements logistic regression to classify spam/ham data set.The data set spambase-train.csv and spambase-test.csv is originally from https://archive.ics.uci.edu/ml/datasets/Spambase .Only 54 features viz the frequency of Words/characters are considered.

### Formulas:
The probability of spam/ham is calculated as follows:

![probablity](https://user-images.githubusercontent.com/43190668/48684176-212bff80-eb76-11e8-820f-e7293093cd32.png)
where P(Y=0) represents ham email.

For simplicity we can set the bias term w0 and an extra variable that is always 1 for our dataset.

![prob](https://user-images.githubusercontent.com/43190668/48684310-af07ea80-eb76-11e8-9c99-8bdb48a1ff7f.png)

We can either maximize the Log-Likelihood or minimize cost function. This program implements maximization of  Log-Likelihood ie gradient ascent.

![log-likelihood](https://user-images.githubusercontent.com/43190668/48684658-4883cc00-eb78-11e8-9abe-450dfec1567e.png)

The gradient for above log-likelihood is :
![gradient](https://user-images.githubusercontent.com/43190668/48684520-90eeba00-eb77-11e8-82cb-cb7ce869e1bd.png)

Where yi is the label ,xij is j-th feature of i-th datapoint.

Thus the parameter vector is updated as follows.

![parameter_update](https://user-images.githubusercontent.com/43190668/48684544-baa7e100-eb77-11e8-9922-c46d425728fc.png)

where η is learning rate.






