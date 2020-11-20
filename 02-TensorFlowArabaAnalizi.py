import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import keras
from keras.layers import Dense,Activation,Dropout
from keras.models import Sequential

dataFrame = pd.read_excel("merc.xlsx")#excel dosyasındaki datamızı getirdik.
print(dataFrame.head())#head ilk 5 kolonu yazdırır
print(dataFrame.describe())
print(dataFrame.isnull().sum()) #datada boş veri varmı ?
sbn.distplot(dataFrame["price"])#dağılım grafiği
plt.show()
sbn.countplot(dataFrame["year"])#değer grafiği
plt.show()
print(dataFrame.corr())
print(dataFrame.corr()["price"].sort_values())#korelasyonlar arası fiyata göre en düşük ilişkiden en yüksek ilişkiyi diz.
sbn.scatterplot(x="mileage",y="price",data=dataFrame)
plt.show()
yuksek20araba = dataFrame.sort_values("price",ascending=False).head(20) #ascending false diyerek büyükten küçüğe sıraladık.head(20) ile ilk 20 tanesini seçtik
print(yuksek20araba)
#dataframe de yüksek fiyatlı cok araba olduğu için verinin iyiliği için %1 lik kısmı silicez fiyatı yüksek arabaların.
print(len(dataFrame))#kaç datamız olduğunu gördük.
print(len(dataFrame) * 0.01) #datamızdaki %1 lik kısmı gördük
yuzde99data = dataFrame.sort_values("price",ascending=False).iloc[131:]#131 den sonrasını data olarak edindik.
print(yuzde99data.describe())#verilerin analizini aldık
sbn.distplot(yuzde99data["price"])
plt.show()
print(yuzde99data.groupby("year").mean()["price"])#yılların ortalamasını fiyat ile yazdır.
dataFrame = yuzde99data
dataFrame = dataFrame[dataFrame != 1970]#1970 yılındaki arabaları cıkarttık.
print(dataFrame.groupby("year").mean()["price"])
dataFrame= dataFrame.drop("transmission",axis=1)#transmission kolonunu sildim

y= dataFrame["price"].values #fiyatı y ye eşitledik
x= dataFrame.drop("price",axis=1).values #fiyatı eğiteceğimiz datadan silerek x e eşitledik
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size =0.33,random_state = 20)#test ve trainleri böldük

scaler = MinMaxScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)


model = Sequential()

model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")


model.fit(x=x_train, y = y_train,validation_data=(x_test,y_test),batch_size=250,epochs=500)
#verilerin hepsini birden vermemiz katmanları yorar. o yüzden batch size 250 diyerek verileri parti parti veriyoruz.
kayipVerisi = pd.DataFrame(model.history.history)
print(kayipVerisi.head())
kayipVerisi.plot()
plt.show()
tahminDizisi = model.predict(x_test)
print(tahminDizisi)
plt.scatter(y_test,tahminDizisi)
plt.plot(y_test,y_test,"g-*")
plt.show()
print(dataFrame.iloc[2])

yeniArabaSeries = dataFrame.drop("price",axis=1).iloc[2]
yeniArabaSeries = scaler.transform(yeniArabaSeries.values.reshape(-1,5))
tahmin = model.predict(yeniArabaSeries)
print(tahmin)