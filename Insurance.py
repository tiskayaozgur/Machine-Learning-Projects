
import numpy as np
import pandas as pd

#Once datasetımızı okuduk
df = pd.read_csv('insurance.csv')
print(df)
print()

#Daha sonra datasetımızde bulunan kategorık verılerı sayısal bıcıme tek hamlede donusturmek ıcın OrdinalEncoder sınıfını kullandık. Bunun yerıne datamızı pandasta okurken read_csv methodunun converters parametresını kullanarak da kategorık verılerı ıstesek sayısal bıcıme donusturebılırdık.
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
oe = OrdinalEncoder()
df[['sex', 'smoker', 'region']] = oe.fit_transform(df[['sex', 'smoker', 'region']])
# print(df)
print()

#Daha sonra sayısal hale gelmıs olan kategorık verılerımıze ohe donusumu yaptık kı makına buyukluk kucukluk ılıskısı algılamasın dıye.
ohe = OneHotEncoder(sparse=False)
ohe_data = ohe.fit_transform(df[['sex', 'smoker', 'region']].to_numpy())
print()


#ohe sonucu elde ettıgımız datalarımızı asıl df'ımıze ınsert edecegız, tabı bunun ıcın df ıcerısınde bulunan asıl datamızı sılmemız gerekmetedır. Once sıldık, daha sonra gereklı yerlerı ınsert ettık.
df.drop(['sex', 'smoker', 'region'], axis=1, inplace=True)
df.insert(1, 'Female', ohe_data[:,0])
df.insert(2, 'Male', ohe_data[:, 1])
df.insert(5, 'Smoker_NO', ohe_data[:, 2])
df.insert(6, 'Smoker_YES', ohe_data[:, 3])
df.insert(7, 'ne', ohe_data[:, 4])
df.insert(8, 'nw', ohe_data[:, 5])
df.insert(9, 'se', ohe_data[:, 6])
df.insert(10, 'sw', ohe_data[:, 7])
print(df)
print()


#Daha sonra datasetımızı ayırdık gırdıler ve cıktılar olacak sekılde
dataset_x = df.iloc[:, :11]
dataset_y = df.iloc[:, 11]

#Daha sonra gırdı kolondakı degerlerımızın hıstını cızdırdık ve duzgun dagılım olmadıgını gorduk, MinMax Scaling uygulayacagız
import pandas as pd
df=pd.DataFrame(dataset_x)
df.hist(figsize=(10,5))
import matplotlib.pyplot as plt
plt.show()


from sklearn.model_selection import  train_test_split
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

