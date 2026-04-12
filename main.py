import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('Data/OnlineNewsPopularity.csv')

df.head()

#sayısal değerler için ayrı tanımlama
num_var= [col for col in df.columns if df[col].dtype != 'O'] #object'in o su

df.shape

df.columns
#isimlerin başlarında boşluklar vardı, strip ile bunları temizledim
df.columns=df.columns.str.strip()

df.info()
#target değeri integer ve predictive değerler float numerik ,url ve timedelta değerlerini sonrasında droplayacağız
#bütün data non-null. Yine de emin olalım
df.isnull().values.any()
#np.False_ aldık. Devam

desc_sum=df.describe().T

#Target incelemesi
df[df.shares>df.shares.mean()].shares.count() #np.int64(8079) -- ortalama üstünde 8079 kayıt var
df[df.shares<df.shares.mean()].shares.count() #np.int64(31565) -- ortalama altında 31.565 kayıt var, verinin oldukça çarpık olduğunu görebiliyoruz, bu bazı makalelerin(outlier) fazlasıyla popüler olmasıyla ilgili.

df.loc[df.shares>df.shares.mean(),'url'] #ortalama üstü paylaşım alan makalelerin linkleri

predictive =df.iloc[:,2:60] #non-predictive ve target fetureları çıkarıldı.(index 0,1,60)
predictive
predictive.columns

from matplotlib import pyplot as plt

sns.boxplot(x= predictive['n_tokens_title'])
plt.show()

##

#veri dağılımının numerik özeti
def num_summary(data, numerical_col, plot=False):
    quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]  # aykırı değer yok
    print(data[numerical_col].describe(quantiles).T)

    if plot:
        data[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(data=predictive, numerical_col="n_tokens_title", plot=True)
#örnek olarak başlık tokenlerinde uyguladım


#bütün sütunlar için uyguluyoruz
#for col in predictive:
 #   num_summary(predictive,col, plot=True)

#
df.groupby('shares')['n_tokens_title'].mean()

#tüm değişkenler için yapalım
def target_summary_with_num(dataframe,target,num_col):
    print(dataframe.groupby(target).agg({num_col: 'mean'}),end="\n\n")

for col in predictive:
    target_summary_with_num(df,'shares',col)


#
#korelasyonlara bakıyoruz
corr = predictive.corr()

#ısı haritası
sns.heatmap(corr,cmap='coolwarm')
plt.show()

corr_matrix = corr.abs()

upper_triangle_matrix= corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype("bool"))
#simetrik olduğu için matrisin içinden bir üçgen aldık.
drop_list= [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
corr_matrix[drop_list]


df= df.drop(drop_list,axis=1)
predictive = predictive.drop(drop_list,axis=1)
#yogun korele değişkenleri silme işlemi, bu verisetinde böyle bir durum olmadığı için silinen değişken yok

#Modeling
#Prediction
#Evaluation
#Hyperparameter Optimisation
#Finalisation

#Roadmap bu şekilde

from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

x=predictive
y=df.shares


#feature scaling yapacağız
x_scaled = StandardScaler().fit_transform(x)
x_scaled_v1 = pd.DataFrame(x_scaled, columns=x.columns)

#PCA NEDİR
#column gurupları kampanya, demografik , "BİLEŞEN ANALİZİ"
#satır bazlı gruplar özel günler vb. "KÜMELEME"
kmeans= KMeans()
elbow = KElbowVisualizer(kmeans,k=(2,20)) #"2 ve 20 arasındaki en uygun bölünecek sayıyı bul"
#kaça böleceğimize bakıyoruz optimum küme sayısını veriyor NASI ÇALIŞIYOR TEKRAR ARASTIR *** ** ** * **

#birbirine yakın en az sayıda küme bulmak daha değerli , cluster sayısı mümkün oldukça az olmalı
elbow.fit(x_scaled_v1)
elbow.elbow_value_

kmeans = KMeans(n_clusters=elbow.elbow_value_,random_state=17).fit(x_scaled_v1)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_


clusters =kmeans.labels_
predictive=predictive.copy()
predictive["cluster"]= clusters

predictive. columns

predictive. shape

predictive.groupby('cluster').agg(["count", "mean", "median"])

predictive.to_csv("clusters.csv")
from sklearn.decomposition import PCA
pca = PCA(n_components=elbow.elbow_value_)  #pca, şu kadar sayıda değişken senin  verini anlatır verisini bize verecek elbowdan elde ettiğimiz k değerini koyduk
pca_fit =pca.fit_transform(x_scaled_v1)

pca.explained_variance_ratio_  #açıklanan varyans oranı

np.cumsum(pca.explained_variance_ratio_)  #bu clusterların etkileme ağırlığı

#sonrası test


#8 bileşenin 2-3 tanesi modeli açıklıyor gibi görünüyor
pca_df = pd.DataFrame(pca_fit,columns=[f"PC{i + 1}" for i in range(pca_fit.shape[1])])
pca_df.index = x.index
#[pca ,1 pca 2, pca 3 ,pca4 ]gibi -> hoca içeri böyle 8 e kadar yazmıştı

final_df = pd.concat([x, pca_df], axis=1)

## PCA İLE LİNREG
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# PCA featurelar ile model
X_pca = pca_df
y = df['shares']

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

pca_model = LinearRegression().fit(X_train, y_train)

# test performansı
from sklearn.metrics import mean_squared_error, r2_score

y_pred = pca_model.predict(X_test)

print("PCA Model RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("PCA Model R2:", r2_score(y_test, y_pred))
##

#basic regresyon modeli örneği
#train ve test olarak ikiye ayrılmalı
df_train = pd.read_csv('Data/OnlineNewsPopularity.csv')

X = df.drop('url',axis=1)

#preprocessing x text y test x train y train ayırma ve column droplama yaptı hoca, kaçırdım
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df.drop(['url', 'shares','timedelta'], axis=1)
y = df['shares']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model_df = LinearRegression().fit(X_train, y_train)

# tahmin
y_pred_lr = reg_model_df.predict(X_test)

print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))

# model parametreleri
reg_model_df.intercept_
reg_model_df.coef_

#intercept çözülemeyen kısmın ağırlığı demektir , ne kadar azsa o kadar iyi

#coefficient -> bir değerin sonucu ne kadar etkilediği

#bias , sabit
#reg_model_df.intercept_
#reg_model.intercept_
#hocanın verisetinde birinin intercepti daha düşük /açıklanabiklirliliği daha yüksek

#coefficient,

#lineer vs decision tree

#cart algoritmasıyla uygulanan decision treede scalingsiz ve pcasiz hata payına bakıyoruz

#GridSearch cv ile decisiontree greressorda hiperparametre optimizasyonu
 #cv(cross validate 5  dedik yani veriyi 5 parçaya böldük ,n_jons=-1, verbose=True .fitx,y
 #buradan best_params aldık
# Decision Tree Regressor (CART)
from sklearn.tree import DecisionTreeRegressor

cart_model = DecisionTreeRegressor(random_state=42)
cart_model.fit(X_train, y_train)

# Tahmin
y_pred_cart = cart_model.predict(X_test)

print("CART RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_cart)))
print("CART R2:", r2_score(y_test, y_pred_cart))


# HYPERPARAMETER OPTIMIZATION
from sklearn.model_selection import GridSearchCV

cart_params = {
    "max_depth": [3, 5, 8, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10]
}

cart_grid = GridSearchCV(
    cart_model,
    cart_params,
    cv=5,
    n_jobs=-1,
    verbose=True
)

cart_grid.fit(X_train, y_train)

# en iyi parametreler
print("Best Params:", cart_grid.best_params_)

# final model
cart_final = DecisionTreeRegressor(**cart_grid.best_params_, random_state=42)
cart_final.fit(X_train, y_train)

# final tahmin
y_pred_final = cart_final.predict(X_test)

print("Final CART RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_final)))
print("Final CART R2:", r2_score(y_test, y_pred_final))

# MODEL KARŞILAŞTIRMA
print("\n--- MODEL COMPARISON ---")
print("Linear Regression R2:", r2_score(y_test, reg_model_df.predict(X_test)))
print("PCA Regression R2:", r2_score(y_test, pca_model.predict(X_test)))
print("CART R2:", r2_score(y_test, y_pred_final))

# FEATURE IMPORTANCE (Decision Tree)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": cart_final.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importance.head(10))




