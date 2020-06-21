import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pylab import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline


path ='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
print(df.dtypes)

df.insert(2,'copy', df[''])

im = LinearRegression()
Z = df[['horsepower', 'highway-mpg', 'curb-weight']]
x = df[['highway-mpg']]
y = df['price']
im.fit(Z, y)
y_hat = im.predict(Z)
# predicting through pipeline
Input = [('scale', StandardScaler()), ('model', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(Z, y)
x3 = pipe.predict(Z)

ni = np.arange(1, 100, 1).reshape(1, -1)
im.fit(x, y)


def polly(model, independent_v, dependant_v, name):
    x_new = np.linspace(15, 50, 100)
    y_new = model(x_new)
    plt.plot(independent_v, dependant_v, '.', x_new, y_new, '-')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.show()

# polynomial regression
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

subplot(2, 2, 3)
dp = sns.distplot(df['price'], hist=False, color='r', label='Actual value')
sns.distplot(y_hat, hist=False, color='b', label='fitted values', ax=dp)
subplot(2, 2, 2)
sns.distplot(df['price'], rug=False, hist=False)
subplot(2, 2, 1)
sns.regplot(df['highway-mpg'], df['price'], data=df)
subplot(2, 2, 4)
sns.distplot(df['price'], bins=2, kde=False)
print(df['price'].value_counts())


k1 = df[['drive-wheels', 'body-style', 'price']]
k2 = k1.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
pivot = k1.pivot(index='drive-wheels', columns='body-style')
pivot1 = pivot.fillna(0)
print(pivot1)
# extracting subgroups
gp2 = k1.groupby(['drive-wheels'])
h1 = gp2.get_group('4wd')['price']
h2 = gp2.get_group('rwd')['price']
x, y = stats.f_oneway(h1, h2)
print('results are: F = ', x,'P = ', y)

# heatmap visualization
fig, ax = plt.subplots()
im = ax.pcolor(pivot1, cmap='RdBu')

#label names
row_labels = pivot1.columns.levels[1]
col_labels = pivot1.index

# move ticks and labels to the center
ax.set_xticks(np.arange(pivot1.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(pivot1.shape[0]) + 0.5, minor=False)

# insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

# rotate label if too long
plt.xticks(rotation=45)
fig.colorbar(im)
plt.show()
