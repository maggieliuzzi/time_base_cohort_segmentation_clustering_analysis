#### Recency, Frequency, Monetary Value and Tenure Segmentation

import pandas as pd
import numpy as np
import datetime
import seaborn as sns
from matplotlib import pyplot as plt




df = pd.read_csv('./BI Challenge Data (1).csv')
print(df.head())
print(df.tail())
# print(df.info())
# print(df.dtypes)
# for col in df:
    # print (type(df[col][1]))
print(df.describe())

# Understand the time span of the dataset
print('Min createDate:{}; Max createDate:{}'.format(min(df.createDate),
                              max(df.createDate)))

# Understand how many posters created completed tasks in this period
print(df['posterID'].nunique())




''' Perform Business Logic Checks | Handle Missing & Invalid Values '''

# print(len(df.index))

# Remove all rows that have at least one missing value
df = df.dropna()

# Business Rule: CreateDate <= AssignedDate <= CompletedDate
df = df.drop(df[(df.createDate > df.assignedDate) | (df.createDate > df.completedDate) | (df.assignedDate > df.completedDate)].index)

# Business Rule: TaskID, WorkerID, PosterID >= 0
df = df.drop(df[(df.taskID < 0) | (df.workerID < 0) | (df.posterID < 0)].index)

# Business Rule: BidCount >= 1
df = df.drop(df[(df.bidCount < 1)].index)

# Business Rule: PostPrice, AssignedPrice >= 5 & <= 9,999
df = df.drop(df[(df.postPrice < 5) | (df.postPrice > 9999) | (df.assignedPrice < 5) | (df.assignedPrice > 9999)].index)




# Calculating difference between assignedPrice and postPrice and assigning it to a new column for future analysis
df['diffPrice'] = df.assignedPrice - df.postPrice




''' Time-Based Cohort Analysis '''

# Set the DataFrame index (row labels) using one or more existing columns.
df.set_index('posterID', inplace=True)

# Divide posters into cohort groups based on the date they created their first completed task
df['CohortGroup'] = df.groupby(level=0)['createDate'].min()
df.reset_index(inplace=True)

user_tot = df.groupby(['createDate', 'posterID']).sum()
user_tot.reset_index(inplace=True)
# print(user_tot.head())

cohorts = df.groupby(['CohortGroup', 'createDate']).agg({'posterID': pd.Series.nunique})

# Make column names more meaningful
cohorts.rename(columns={'posterID': 'Total Cohort Posters in Period','createDate': 'Next Months', 'CohortGroup': 'Cohort Group'}, inplace=True)
# print(cohorts.head())

def cohort_period(df):
    df['CohortPeriod'] = np.arange(len(df)) + 1
    return df

cohorts = cohorts.groupby(level=0).apply(cohort_period)

# Re-index df
cohorts.reset_index(inplace=True)
cohorts.set_index(['CohortGroup', 'CohortPeriod'], inplace=True)

# Create a Series holding total size of each CohortGroup
cohort_group_size = cohorts['Total Cohort Posters in Period'].groupby(level=0).first()

# Divide all values in cohorts table by cohort_group_size
user_retention = cohorts['Total Cohort Posters in Period'].unstack(0).divide(cohort_group_size, axis=1)
user_retention_abs = cohorts['Total Cohort Posters in Period'].unstack(0)
# print(user_retention.head(10))
# print(user_retention_abs.head(10))




''' Plotting and analysing cohort behaviour '''

width, height = plt.figaspect(4)
fig = plt.figure(figsize=(width, height), dpi=400)
from matplotlib import rcParams

plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'DejaVu Sans'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 6
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 6
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
# plt.show()

# 1
width, height = plt.figaspect(.6)
fig = plt.figure(figsize=(width, height), dpi=120)
plt.title("Poster Cohort Analysis (abs)", fontname='Ubuntu', fontsize=14, fontweight='bold')
sns.heatmap(user_retention_abs.T, mask=user_retention.T.isnull(), annot=True, fmt='g', cmap='coolwarm')
# plt.show()

# 2
width, height = plt.figaspect(.6)
fig = plt.figure(dpi=120)
plt.title("Churn Analysis", fontname='Ubuntu', fontsize=14, fontweight='bold')
sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='0.00%', cmap='viridis')
# plt.show()

# 3
width, height = plt.figaspect(.6)
fig = plt.figure(dpi=150)

ax = user_retention[['2013-06', '2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12',
                     '2014-01','2014-02','2014-03','2014-04','2014-05','2014-06','2014-07','2014-08','2014-09','2014-10','2014-11','2014-12',
                     '2015-01','2015-02','2015-03','2015-04','2015-05','2015-06','2015-07','2015-08','2015-09','2015-10','2015-11','2015-12',
                     '2016-01','2016-02','2016-03','2016-04','2016-05','2016-06','2016-07','2016-08','2016-09','2016-10','2016-11']].plot(figsize=(11, 6))
plt.title("Retention rate (%) per CohortGroup", fontname='Ubuntu', fontsize=20, fontweight='bold')

plt.xticks(np.arange(1, 16.1, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
ax.set_xlabel("CohortPeriod", fontsize=10)
ax.set_ylabel("Retention(%)", fontsize=10)
plt.savefig("cohort_retention.png")
# plt.show()

# 4
ax = user_retention.T.mean().plot(figsize=(11, 6), marker=',')
plt.title("Retention rate (%) per CohortGroup", fontname='Ubuntu', fontsize=6, fontweight='bold')

plt.xticks(np.arange(1, 42.1, 1), fontsize=4)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=4)
ax.set_xlabel("CohortPeriod", fontsize=4)
ax.set_ylabel("Retention(%)", fontsize=4)
# plt.show()




''' RFMT | Percentile-based Grouping '''

# Sort posters based on each metric
# Divide posters into a pre-defined number of groups
# Assign a label to each group

# Create a spend quartile with 4 groups and labels ranging from 1 to 4
spend_quartile = pd.qcut(df['assignedPrice'], q=4, labels=np.arange(1,5))

# Add 'Spend_Quartile' column with quartile values to df
df['AssignedPrice_Quartile'] = spend_quartile
print(df.sort_values('AssignedPrice_Quartile'))

# Setting snapshot date to 1st Dec 2016
df['createDate'] = pd.to_datetime(df['createDate'])
snapshot_date = max(df.createDate) + datetime.timedelta(days=1)
snapshot_date = pd.to_datetime(snapshot_date)

# Extract difference in months from all previous values
years_diff = snapshot_date.year - df['createDate'].dt.year
months_diff = snapshot_date.month - df['createDate'].dt.month
df['Recency_Months'] = years_diff * 12 + months_diff
print(df.sort_values('Recency_Months'))

# Store recency labels from 4 to 1 in decreasing order
r_labels = list(range(4, 0, -1))
# Create a spend quartile with 4 groups and pass the previously created labels
recency_groups = pd.qcut(df['Recency_Months'], q=4, labels=r_labels)

# Calculate Recency, Frequency and Monetary Value of each poster
RFMT_df = df.groupby(['posterID']).agg({
    'createDate': 'count',
    'Recency_Months': {'Recency_Months': 'min', 'Tenure_Months': 'max'},
    'assignedPrice': 'sum'})

# Eg. Profit: 5% commission per dollar paid by poster
RFMT_df['assignedPrice'] = RFMT_df['assignedPrice']*0.05

# Rename the columns
RFMT_df.rename(columns={'createDate': 'Frequency_Count', # Over period: 2013-06 - 2016-11
                       'assignedPrice': 'MonetaryValue_TotalProfit'}, inplace=True)

# Dropping posters with more than 25 completed task (considered outliers based on previous analysis)
# Before: Min Frequency_Count:1; Max Frequency_Count:1143, i.e. at least one poster seemed to have 1143 completed tasks.
RFMT_df = RFMT_df.drop(RFMT_df[RFMT_df['Frequency_Count']['count'] > 25.0].index)


# Create a recency quartile with 4 groups and pass the previously created labels
recency_groups = pd.qcut(RFMT_df['Recency_Months']['Recency_Months'], q=4, labels=r_labels)

# Tenure: number of months since a poster created their first completed task
t_labels = range(1, 5)
tenure_groups = pd.qcut(RFMT_df['Recency_Months']['Tenure_Months'], q=4, labels=t_labels)

frequency_groups = pd.cut(RFMT_df['Frequency_Count']['count'], bins=[0,1,6,25,1000000], labels=[1,2,3,4])

m_labels = range(1, 5)
# Assign these labels to 4 equal percentile groups
m_groups = pd.qcut(RFMT_df['MonetaryValue_TotalProfit']['sum'], q=4, labels=m_labels)

# Add 'Monetary Value', 'Recency', 'Tenure' and 'Frequency' columns to df
RFMT_df = RFMT_df.assign(MonetaryValue=m_groups)
RFMT_df = RFMT_df.assign(Recency=recency_groups.values, Tenure=tenure_groups.values, Frequency=frequency_groups.values)



RFMT_df['RFMT_Score'] = RFMT_df[['Recency','Frequency','MonetaryValue','Tenure']].sum(axis=1)

print('Min RFMT_Score:{}; Max RFMT_Score:{}'.format(min(RFMT_df['RFMT_Score']),
                              max(RFMT_df['RFMT_Score'])))

RFMT_groupByScore = RFMT_df.groupby('RFMT_Score').mean()
# print(RFMT_groupByScore)



def rfmt_level(df):
    if df >= 10: # df = df[RMFT_Score]
        return 'Gold'
    elif (df >= 8) and (df < 10):
        return 'Silver'
    elif (df >= 6) and (df < 8):
        return 'Bronze'
    else:
        return 'Basic'

RFMT_df['RFMT_Level'] = RFMT_df['RFMT_Score'].apply(rfmt_level)

RFMT_df['Recency'] = pd.to_numeric(RFMT_df['Recency'])
RFMT_df['Frequency'] = pd.to_numeric(RFMT_df['Frequency'])
RFMT_df['MonetaryValue'] = pd.to_numeric(RFMT_df['MonetaryValue'])
RFMT_df['Tenure'] = pd.to_numeric(RFMT_df['Tenure'])

# Calculate average recency, frequency, tenure and monetary value for each RFMT_Level
RFMT_level_agg = RFMT_df.groupby(['RFMT_Level']).agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    # Return the size of each segment
    'Tenure': 'mean',
    'MonetaryValue': ['mean', 'count']
})
print(RFMT_level_agg)
# print(RFMT_level_agg.mean())
# print(RFMT_level_agg.std())




''' Customer Lifetime Value (CLV) '''

from lifetimes import BetaGeoFitter

bgf = BetaGeoFitter()
bgf.fit(RFMT_df['Frequency_Count']['count'], RFMT_df['Recency_Months']['Recency_Months'], RFMT_df['Recency_Months']['Tenure_Months'])
# print(bgf)

# Compute number of completed tasks posters are expected to create in the next time period, given their recency and frequency
from lifetimes.plotting import plot_frequency_recency_matrix
plot_frequency_recency_matrix(bgf)
# plt.show()

# Compute probability of posters being active in the next time period
from lifetimes.plotting import plot_probability_alive_matrix
plot_probability_alive_matrix(bgf)
# plt.show()

# Compute number of completed tasks a poster is expected to create in the next period, given their frequency, recency and tenure
t = 1
RFMT_df['Predicted_Completed_Tasks'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, RFMT_df['Frequency_Count']['count'], RFMT_df['Recency_Months']['Recency_Months'], RFMT_df['Recency_Months']['Tenure_Months'])
print(RFMT_df.sort_values(by='Predicted_Completed_Tasks').round(2))

# Assess model fitness
from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf)
# plt.show()

# This library can also be used to calculate frequency, recency and tenure directly from the original df
from lifetimes.utils import summary_data_from_transaction_data
# print(df.head())
summary = summary_data_from_transaction_data(df, 'posterID', 'createDate', observation_period_end='2016-11-30')
# print(summary.head())
bgf.fit(summary['frequency'], summary['recency'], summary['T'])

# Predict number of completed tasks in the next 10 time periods for a specific poster
t = 10
individual = summary.iloc[4] # 20: 0.0032530228448847194, 4: 0.00012172605810732591
# The below function is an alias to `bfg.conditional_expected_number_of_purchases_up_to_time
print(bgf.predict(t, individual['frequency'], individual['recency'], individual['T']))

# Calculate historical probability of a poster being active, given their transaction history and trained model
from lifetimes.plotting import plot_history_alive
posterID = 4
tenure = 38
RFMT_df = RFMT_df.reset_index() # ensure nothing is affected
sp_trans = RFMT_df.loc[RFMT_df['posterID'] == posterID]
# plot_history_alive(bgf, tenure, sp_trans, 'createDate') # Debug: (KeyError: 'createDate')
# plt.show()

# Estimate Customer Lifetime Value (CLV) using Gamma-Gamma model
# Test model assumption: no relationship between monetary value and transaction frequency (i.e. correlation close to 0)
print(RFMT_df[['MonetaryValue_TotalProfit', 'Frequency_Count']].corr()) # 0.73 in this case
from lifetimes import GammaGammaFitter
ggf = GammaGammaFitter(penalizer_coef = 0)
ggf.fit(RFMT_df['Frequency_Count']['count'], RFMT_df['MonetaryValue_TotalProfit']['sum'])
# print(ggf)
print(ggf.conditional_expected_average_profit(RFMT_df['Frequency_Count']['count'],RFMT_df['MonetaryValue_TotalProfit']['sum']).round(2).head(10))
print("Expected conditional average profit: %s, Average profit: %s" % (ggf.conditional_expected_average_profit(RFMT_df['Frequency_Count']['count'],RFMT_df['MonetaryValue_TotalProfit']['sum']).mean().round(2),RFMT_df[RFMT_df['Frequency_Count']['count']>0]['MonetaryValue_TotalProfit']['sum'].mean().round(2)))


# Compute total CLV using DCF method (https://en.wikipedia.org/wiki/Discounted_cash_flow), adjusting for cost of capital
# Re-fit model to RFMT_df
bgf.fit(RFMT_df['Frequency_Count']['count'], RFMT_df['Recency_Months']['Recency_Months'],RFMT_df['Recency_Months']['Tenure_Months'])
# bgf: model that predicts number of future transactions | time unit: months | monthly discount rate ~ 12.7% annually
print(ggf.customer_lifetime_value(bgf, RFMT_df['Frequency_Count']['count'],RFMT_df['Recency_Months']['Recency_Months'],RFMT_df['Recency_Months']['Tenure_Months'],RFMT_df['MonetaryValue_TotalProfit']['sum'],time=12,discount_rate=0.01).round(2).head(10))




''' Preprocessing for K-Means clustering '''

# K-Means assumes:
# Variables are normally distributed (not skewed)
# Variables have same average values
# Variables have same variance and standard deviation
# Round clusters (non-elongated data)

print(RFMT_df.describe().round(2))

f, axes = plt.subplots(3, 1, sharex=True, sharey=True)
# Plot histogram with bin size determined automatically
sns.distplot(RFMT_df['Recency'], ax=axes[0])
sns.distplot(RFMT_df['Frequency'], ax=axes[1])
sns.distplot(RFMT_df['MonetaryValue'], ax=axes[2])
# plt.show()

RFMT_df = RFMT_df.drop(['Recency_Months', 'Frequency_Count', 'MonetaryValue_TotalProfit'], axis=1)


# Unskew variables using logarithmic transformation (works only on positive values)

RFMT_df_log = pd.DataFrame(np.log(RFMT_df['Recency']))
RFMT_df_log = RFMT_df_log.assign(Frequency = np.log(RFMT_df['Frequency']))
RFMT_df_log = RFMT_df_log.assign(MonetaryValue = np.log(RFMT_df['MonetaryValue']))

f, axes = plt.subplots(3, 1, sharex=True, sharey=True)
sns.distplot(RFMT_df_log['Recency'], ax=axes[0])
sns.distplot(RFMT_df_log['Frequency'], ax=axes[1])
sns.distplot(RFMT_df_log['MonetaryValue'], ax=axes[2])
# plt.show()


# Normalise -center and scale- variables (since K-Means works better on variables with the same mean, variance and st dev)
# z = (x - u) / s, where u: mean, s: standard deviation
# data_normalised = (data - data.mean()) / data.std()
# Returns numpy.ndarray object, which will make K-Means run faster
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(RFMT_df_log)
RFMT_df_log_norm = scaler.transform(RFMT_df_log)
RFMT_df_log_norm = pd.DataFrame(data=RFMT_df_log_norm, index=RFMT_df_log.index, columns=RFMT_df_log.columns)

print('mean: ', RFMT_df_log_norm.mean(axis=0).round(2))
print('std: ', RFMT_df_log_norm.std(axis=0).round(2))
print(RFMT_df_log_norm.describe().round(2))




''' K-Means Clustering (Unsupervised Learning) '''

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=1)

kmeans.fit(RFMT_df_log_norm)
# Extract cluster labels from labels_ attribute and store them as cluster_labels object.
cluster_labels = kmeans.labels_

RFMT_k4 = RFMT_df.assign(Cluster = cluster_labels)




## Choose number of clusters

# Elbow criterion method
# Plot number of clusters against within-cluster sum-of-squared-errors (SSE) - sum of squared distances from every data point to their cluster center
# Identify an "elbow" in the plot, i.e. a point representing an "optimal" number of clusters. Point where diminishing returns of adding one more cluster start.
# Also consider business needs
# Also test with other n_clusters

# Fit KMeans and calculate SSE for each k
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(RFMT_df_log_norm)
    # Assign sum of squared distances to closest cluster center to k element of sse dictionary
    sse[k] = kmeans.inertia_
# Plot the sum of squared errors for each value of k and identify if there is an elbow
plt.title('The Elbow Method')
plt.xlabel('k'); plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
# plt.show()




''' Creating poster personas '''

## View summary statistics for each cluster

# Average RFMT values:
RFMT_Clusters = RFMT_k4.groupby(['Cluster']).mean().round(2)
print(RFMT_Clusters)


## Use snake plots to understand and compare segments

# Plot each cluster's average normalised values for each attribute
RFMT_df_log_norm = pd.DataFrame(RFMT_df_log_norm, index=RFMT_df.index, columns=RFMT_df.columns)
RFMT_df_log_norm['Cluster'] = RFMT_k4['Cluster']

# Transform the normalised RFMT data into a long format by "melting" the metric columns into two columns - one for name of metric and one for actual numeric value
df_melt = pd.melt(RFMT_df_log_norm.reset_index(),
                    id_vars=['posterID', 'Cluster'],
                    value_vars=['Recency', 'Frequency', 'MonetaryValue'],
                    var_name='Metric', # or 'Attribute'
                    value_name='Value')
print(df_melt.head())
print(df_melt.tail())

plt.title('Snake plot of standardised/normalised variables')
plt.xlabel('Metric')
plt.ylabel('Value')
sns.lineplot(data=df_melt, x="Metric", y="Value", hue='Cluster')
# plt.show()


## Analyse relative importance of cluster attributes compared to overall population average

# Calculate importance score by dividing average values for each cluster by average population values, then subtracting 1
# The further a ratio is from 0, the more important that attribute is for a segment relative to the total population
cluster_avg = RFMT_k4.groupby(['Cluster']).mean()
population_avg = RFMT_df.mean()
relative_imp = cluster_avg / population_avg - 1
print(relative_imp.round(2))

plt.figure(figsize=(8, 2))
plt.title('Relative importance of metrics')
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
# plt.show()
