# Recommendation-System (Notes and Summary of Learning and Thinking)

### Latent Factor Model in Recommendation

### Advantages

- Users' personal opinions

LFM methods based on user behaviors (data itself) represent users' personal opinions towards the classification of items. LFM methods have the similar key ideas compared with ItemCF: Two items, if they are clicked, watched, or liked by many users, there's a pretty good chance to classify those two items into same class/genere.

- Decide the number of final classes/generes

We can assign the number of final classes what we want LFM to classify based on user and item vectors.

- The weight of each class/genere

LFM will generate the weight of each class/genere, and thus LFM can learn which class that item belongs to. 'Soft' classification.

- Different dimensions for different classes

According to user profile or user preference, classes computed by LFM are not in the same dimensions.

- Decide the weight of one item in one class

The weight will be large if that item can represent that class (the attribute). For example, if users in one class almost have high chances to click or love one item, that item could have high weight in that class.

#### Latent Factor Model formula: calculate user($u$)'s interest of item($i$)

$\Preference(u, i) = r_{ui} = {p_u}^T q_i = \displaystyle\sum_{f=1}^{F} p_{u,k}q_{i,k}$, where $k$ is $(u,i)$ pair, and $f$ is the indicate of per pair in $k$.

$p_{u,k}$ and $q_{i,k}$ are the parameter of LFM model, $p_{u,k}$ measures the association between user($u$)'s interest and the $k$th latent factor, and $q_{i,k}$ measures the association between the $k$th latent factor and item($i$).

### Approach:

However, in our recommendation system, we do not have negative samples (what kinds of items users are not interested in):

- interested items: user have many actions on those items
- not very interested items: users have a few historical behaviors on those items
- noninterested items: users do not have any actions on them

We considere two approaches here:

##### 1. Randomly generate negative samples:

First, we manually select negative samples for each user
- For every user, we need to keep balance between positive (interested) and negative (noninterested) samples. Hence, we need to make they have similar sample sizes.
- We attempt to collect the hot/top items, and yet users do not have any historical behaviors for a while (one week, or one month).


```python
def NegativeSample(self, item):
    ret = dict()
    for i in item.keys():
        ret[i] = 1
    n = 0
    for i in range(0, len(item) * 5):
        item = itemlist[random.randint(0, len(itemlist)) - 1] 
#itemlist is a table for all items' popolarity (count by number of click) and item show counts
        if item in ret:
            continue
        ret[item] = 0
        n += 1
        if n > len(item):
            break
    return ret
```

Then, if $(u, i)$ is positive sample, $r_{ui}$ = 1, and if $(u, i)$ is negative sample, $r_{ui}$ = -1.

##### 2. Explicit and Implicit rating:

Most matrix factorization focuses on explicit feadback datasets, and the LFM results have relatively good accuracies. Instead, implicit feedback (e.g. browsing history, click-through-rate, payment hisgory, view count, and view duration, etc.) must be very powerful to be used to analyze users' behavior and preference. 

For each user, we know the number of times the user click on public accounts or video, and we also know the number of times they save, or the duration length they watch the video. Thus, we compute the $interest$ including the $weights$ for given user $i$ and item(public account or video link) $i$ to be the user's click/action frequency level of that item and normalized by user's total click or watch:

$interest_{u,i} = \frac{count_{click} (u,i) + count_{save} (u,i) + view_{duration} (u,i)}{\displaystyle\sum_{i'} count_{click} (u,i') + \displaystyle\sum_{i'} count_{save} (u,i') + \displaystyle\sum_{i'} view_{duration} (u,i')}$

Next, we can calculate the rating for user $i$ to item $i$: $r_{u,i} = 5 * (1 - \displaystyle\sum_{i' = 1}^{K - 1} interest_{u, i'})$, and then we could get a rating in 0 - 5 range. We will use the rating $r_{u,i}$ in our matrix factorization model.

#### Matrix Factorization (Cont'd):

Accourding to learning process, we know that those two parameters in Latent Factor Model are learned from training dataset, including user($u$)'s interested items and noninterested items.

$Cost$ $Function$ = $\displaystyle\sum_{(u,i) \in K} (r_{u,i} - \hat{r}_{u,i}) ^ 2 $ = $\displaystyle\sum_{(u,i) \in K} (r_{u,i} - \displaystyle\sum_{k = 1}^{K} p_{u,k}q_{i,k})^2 + \lambda||p_u||^2 + \lambda ||q_i||^2$

The two parts with $\lambda$ are the regularization terms in order to avoid or reduce overfitting. 

Before applying stochastic gradient descent, we first need to take partial derivative for $p_{u,k}$ and $q_{i,k}$ in cost function:

$\frac{\partial C}{\partial p_{u,k}} = -2q_{i,k} + 2\lambda p_{u,k}$

$\frac{\partial C}{\partial q_{i,k}} = -2p_{u,k} + 2\lambda q_{i,k}$

Then, apply stochastic gradient descent by adding learning rate $\gamma$:

$p_{u,k} \leftarrow p_{u,k} + \gamma * (q_{i,k} - \lambda p_{u,k})$

$q_{i,k} \leftarrow q_{i,k} + \gamma * (p_{u,k} - \lambda q_{i,k})$

- first approach according to randomly generate negative sample:


```python
def LFM(user_items, K, N, gamma = 0.0002, lamb = 0.05): # can change the values of gamma and lambda
    [p, q] = startModel(user_items, K)
    for step in range (0, len(N)):
        for user, item in user_items.item():
            sample = NegativeSample(item)
            for newitem, rui in samples.item():
                eui = rui - Predict(user, newitem)
                for k in range(0, len(K)):
                    p[user][k] += alpha * (eui * q[newitem][k] - lamb * p[user][k])
                    p[newitem][k] += alpha * (eui * p[user][k] - lamb * q[newitem][k])
```

- second approach according to compute rui by implicit information:

 $r_{u,i} = 5 * (1 - \displaystyle\sum_{i' = 1}^{K - 1} interest_{u, i'})$


```python
def Recommendation(user, p, q):
    rank = dict()
    for k, puk in p[user].item():
        for i, qki in q[k].item():
            if i not in rank:
                rank[i] += puk * qki
    return rank
```

### Evaluation

#### Offline - RMSE & MAE

$K$ is a set contained all pairs of user and item.

Root Mean Square Error (RMSE): $RMSE = \frac{\sqrt{ \displaystyle\sum_{(u, i) \in K}^{K} (r_{ui} - \hat{r}_{ui})^2}}{|K|}$.

Mean Absolute Value (MAE): $RMSE = \frac{|\displaystyle\sum_{(u, i) \in K}^{K} (r_{ui} - \hat{r}_{ui})^2|}{|K|}$.


```python
import math
import numpy as np

def RMSE(set_user_item):
    return math.sqrt(np.sum([(rui - pui)^2 for u, i, rui, pui in set_user_item]) / float(len(set_user_item)))

def MAE(set_user_item):
    return (np.sum([abs(rui - pui) for u, i, rui, pui in set_user_item]) / float(len(set_user_item)))
```

#### Online - Precision & Recall

$Train(u)$ is the recommendation list generated from LFM mothod under training/historical dataset.

$Test(u)$ is the actual user behaviors on items in real time.

Precision: $\frac{TP}{TP + FP} = \frac{\displaystyle\sum_{u \in U} |Train(u) \cap Test(u)|}{\displaystyle\sum_{u \in U} |Train(u)|}$ = $\frac{good \ movie \ recommended}{all \ good \ movie}$

Recall: $\frac{TP}{TP + FN} = \frac{\displaystyle\sum_{u \in U} |Train(u) \cap Test(u)|}{\displaystyle\sum_{u \in U} |Test(u)|}$ = $\frac{good \ movie \ recommended}{all \ recommended}$


```python
def PrecisionandRecall(test, N):
    #initialize
    hit = 0
    num_recall = 0
    num_precision = 0
    for user, item in test.item():
        rank = Recommendation(user, N)
        hit += len(rank & item)
        num_recall += len(item)
        num_precision += N
    return hit / (1.0 * num_recall), hit / (1.0 * num_precision)
```

