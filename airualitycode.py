!pip install catboost
import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.max_columns', None)
import seaborn as sn
import matplotlib.pyplot as mp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import warnings as w

w.filterwarnings('ignore')

df=pd.read_csv('delhiaqi.csv')
df.head(3)
df.isna().sum()
df.drop('date',axis=1,inplace=True)
df.shape
df.info()
df.duplicated().sum()
sn.boxplot(df)
df['co'].max()
df['co_to_pm2_5'] = df['co'] / df['pm2_5']
df['no_to_pm2_5'] = df['no'] / df['pm2_5']
df['nox'] = df['no'] + df['no2']
df['nox_to_pm2_5'] = df['nox'] / df['pm2_5']
df['total_pollutant'] = df['co'] + df['no'] + df['no2'] + df['o3'] + df['so2'] + df['pm2_5'] + df['pm10'] + df['nh3']
df['avg_pollutant'] = df['total_pollutant'] / 8
df['pm2_5_to_pm10'] = df['pm2_5'] / df['pm10']

df['co_no_interaction'] = df['co'] * df['no']
df['so2_o3_interaction'] = df['so2'] * df['o3']

df['co_squared'] = df['co'] ** 2
df['log_pm2_5'] = np.log1p(df['pm2_5'])

df['high_pm2_5'] = np.where(df['pm2_5'] > 100, 1, 0)
df['high_co'] = np.where(df['co'] > 2000, 1, 0)
df.head()
df.columns
sn.set(style="whitegrid")

# 1. Histogram of PM2.5
mp.figure(figsize=(10, 6))
sn.histplot(df['pm2_5'], bins=10, kde=True)
mp.title('Histogram of PM2.5')
mp.xlabel('PM2.5 Concentration')
mp.ylabel('Frequency')
mp.show()
mp.figure(figsize=(10, 6))
sn.boxplot(x=df['no2'])
mp.title('Boxplot of NO2 Levels')
mp.xlabel('NO2 Concentration')
mp.show()

mp.figure(figsize=(10, 6))
sn.scatterplot(x='pm2_5', y='pm10', data=df)
mp.title('Scatter Plot of PM2.5 vs PM10')
mp.xlabel('PM2.5 Concentration')
mp.ylabel('PM10 Concentration')
mp.show()

mp.figure(figsize=(10, 10))
sn.pairplot(df,hue='pm2_5')
mp.title('Pairplot of Air Quality Features')
mp.show()

mp.figure(figsize=(18, 10))
correlation_matrix = df.corr()
sn.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
mp.title('Correlation Heatmap')
mp.show()

df.corr()

mp.figure(figsize=(10, 6))
sn.histplot(df['co'], bins=10, kde=True)
mp.title('Distribution of CO Levels')
mp.xlabel('CO Concentration')
mp.ylabel('Density')
mp.show()

mp.figure(figsize=(10, 6))
sn.violinplot(x=df['so2'])
mp.title('Violin Plot of SO2 Levels')
mp.xlabel('SO2 Concentration')
mp.show()

mp.figure(figsize=(10, 6))
g = sn.FacetGrid(df, col='high_pm2_5')
g.map(sn.histplot, 'no2')
mp.title('Facet Grid of NO2 Levels by High PM2.5 Indicator')
mp.show()

mp.figure(figsize=(10, 6))
sn.countplot(x='high_pm2_5', data=df)
mp.title('Count of High PM2.5 Indicators')
mp.xlabel('High PM2.5 Indicator')
mp.ylabel('Count')
mp.show()

mp.figure(figsize=(10, 6))
sn.jointplot(x='pm2_5', y='pm10', data=df, kind='scatter', color='g')
mp.title('Joint Plot of PM2.5 vs PM10')
mp.show()

mp.figure(figsize=(10, 6))
sn.histplot(df['nh3'], bins=10, kde=True)
mp.title('Histogram of NH3 Levels')
mp.xlabel('NH3 Concentration')
mp.ylabel('Frequency')
mp.show()

mp.figure(figsize=(10, 6))
sn.boxplot(x='high_pm2_5', y='co', data=df)
mp.title('Boxplot of CO Levels by High PM2.5 Indicator')
mp.xlabel('High PM2.5 Indicator')
mp.ylabel('CO Concentration')
mp.show()

mp.figure(figsize=(10, 6))
sn.kdeplot(df['no2'], shade=True)
mp.title('KDE Plot of NO2 Levels')
mp.xlabel('NO2 Concentration')
mp.ylabel('Density')
mp.show()

mp.figure(figsize=(10, 6))
df['total_pollutant'].plot(kind='area', alpha=0.5)
mp.title('Area Plot of Total Pollutants Over Time')
mp.xlabel('Time Point')
mp.ylabel('Total Pollutant Concentration')
mp.show()

mp.figure(figsize=(12, 8))
sn.heatmap(df[['pm2_5', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3']].corr(), annot=True, cmap='coolwarm')
mp.title('Heatmap of PM2.5 and Other Pollutants')
mp.show()

mp.figure(figsize=(10, 6))
sn.barplot(x=df.index, y='pm2_5_to_pm10', data=df)
mp.title('PM2.5 to PM10 Ratio')
mp.xlabel('Time Point')
mp.ylabel('PM2.5 to PM10 Ratio')
mp.show()

mp.figure(figsize=(10, 6))
sn.scatterplot(x='no', y='no2', data=df)
mp.title('Scatter Plot of NO vs NO2')
mp.xlabel('NO Concentration')
mp.ylabel('NO2 Concentration')
mp.show()

mp.figure(figsize=(8, 8))
df['high_pm2_5'].value_counts().plot(kind='pie', autopct='%1.1f%%')
mp.title('Distribution of High PM2.5 Indicators')
mp.ylabel('')
mp.show()

sn.set(style="whitegrid")

mp.figure(figsize=(18, 15))

mp.subplot(4, 3, 1)
sn.scatterplot(x='co', y='pm2_5', data=df)

mp.title('PM2.5 vs CO')
mp.xlabel('CO Concentration')
mp.ylabel('PM2.5 Concentration')

mp.subplot(4, 3, 2)
sn.scatterplot(x='no2', y='pm2_5', data=df)
mp.title('PM2.5 vs NO2')
mp.xlabel('NO2 Concentration')
mp.ylabel('PM2.5 Concentration')

mp.subplot(4, 3, 3)
sn.scatterplot(x='o3', y='pm2_5', data=df)
mp.title('PM2.5 vs O3')
mp.xlabel('O3 Concentration')
mp.ylabel('PM2.5 Concentration')

mp.subplot(4, 3, 4)
sn.scatterplot(x='so2', y='pm2_5', data=df)
mp.title('PM2.5 vs SO2')
mp.xlabel('SO2 Concentration')
mp.ylabel('PM2.5 Concentration')

mp.subplot(4, 3, 5)
sn.scatterplot(x='pm10', y='pm2_5', data=df)
mp.title('PM2.5 vs PM10')
mp.xlabel('PM10 Concentration')
mp.ylabel('PM2.5 Concentration')

mp.subplot(4, 3, 6)
sn.scatterplot(x='nh3', y='pm2_5', data=df)
mp.title('PM2.5 vs NH3')
mp.xlabel('NH3 Concentration')
mp.ylabel('PM2.5 Concentration')

mp.subplot(4, 3, 7)
sn.boxplot(x='high_pm2_5', y='pm2_5', data=df)
mp.title('Boxplot of PM2.5 by High PM2.5 Indicator')
mp.xlabel('High PM2.5 Indicator')
mp.ylabel('PM2.5 Concentration')

mp.subplot(4, 3, 8)
sn.barplot(x='avg_pollutant', y='pm2_5', data=df)
mp.title('Avg Pollutant vs PM2.5')
mp.xlabel('Avg Pollutant Concentration')
mp.ylabel('PM2.5 Concentration')

mp.subplot(4, 3, 9)
sn.violinplot(y='pm2_5', data=df)
mp.title('Violin Plot of PM2.5')
mp.ylabel('PM2.5 Concentration')

mp.subplot(4, 3, 10)
sn.histplot(df['pm2_5'], bins=10, kde=True)
mp.title('Histogram of PM2.5')
mp.xlabel('PM2.5 Concentration')
mp.ylabel('Density')

# Adjust layout
mp.tight_layout()
mp.show()

x=df.drop('pm2_5',axis=1)
y=df['pm2_5']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

pca = PCA(n_components=14)  # Set number of components to 17
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(14)])

explained_variance = pca.explained_variance_ratio_

for i, variance in enumerate(explained_variance):
    print(f'PC{i+1}: {variance:.4f}')

    mp.figure(figsize=(10, 5))
mp.bar(range(1, 15), explained_variance, alpha=0.7, color='b', align='center')
mp.xlabel('Principal Components')
mp.ylabel('Explained Variance Ratio')
mp.title('Explained Variance Ratio of Principal Components')
mp.xticks(range(1, 15))
mp.show()

pca_df

X_train, X_test, y_train, y_test = train_test_split(pca_df, y, test_size=0.2, random_state=42)

catboost_model = CatBoostRegressor(verbose=0, random_seed=42)
lgbm_model = LGBMRegressor(random_state=42)
ridge_model = Ridge()

stacked_model = StackingRegressor(
    estimators=[('catboost', catboost_model), ('lgbm', lgbm_model)],
    final_estimator=ridge_model,
    cv=3, 
    n_jobs=-1
)
stacked_model.fit(X_train, y_train)

y_pred = stacked_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2:.4f}')

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results

mp.figure(figsize=(12, 6))
mp.plot(results['Actual'].values, label='Actual', color='b', linewidth=2)
mp.plot(results['Predicted'].values, label='Predicted', color='r', linestyle='--', linewidth=2)
mp.title('Actual vs. Predicted Values')
mp.xlabel('Sample Index')
mp.ylabel('PM2.5')
mp.legend()
mp.show()

mp.figure(figsize=(8, 8))
sn.scatterplot(x=results['Actual'], y=results['Predicted'], alpha=0.7, color='purple')
mp.plot([results['Actual'].min(), results['Actual'].max()], [results['Actual'].min(), results['Actual'].max()], color='r', linestyle='--')
mp.title('Actual vs. Predicted Scatter Plot')
mp.xlabel('Actual PM2.5')
mp.ylabel('Predicted PM2.5')
mp.show()

