import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('Data/OnlineNewsPopularity.csv')

df.head()

#sayısal değerler için ayrı tanımlama
num_var= [col for col in df.columns if df[col].dtype != 'O'] #object'in o su

import seaborn as sns
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
for col in predictive:
    num_summary(predictive,col, plot=True)

#
df.groupby('shares')['n_tokens_title'].mean()

#tüm değişkenler için yapalım
def target_summary_with_num(dataframe,target,num_col):
    print(dataframe.groupby(target).agg({num_col: 'mean'}),end="\n\n")

for col in predictive:
    target_summary_with_num(predictive,'shares',col)

#
#korelasyonlara bakıyoruz
corr = predictive.corr()

#ısı haritası
sns.heatmap(corr,cmap='coolwarm')
plt.show()

corr_matrix = corr.abs()

upper_triangle_matrix= corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype("bool"))
#simetrik olduğu için matrisin içinden bir üçgen aldık.
drop_list= [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>90)]
corr_matrix[drop_list]


df= df.drop(drop_list,axis=1)
predictive = predictive.drop(drop_list,axis=1)
#yogun korele değişkenleri silme işlemi, bu verisetinde böyle bir durum olmadığı için silinen değişken yok

