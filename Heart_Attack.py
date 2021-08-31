import numpy as np

#Datasetımızı okuduk.
dataset = np.loadtxt('heart_failure_clinical_records_dataset.csv', delimiter=',', skiprows=1)
print(dataset)
print()

#Datasetımızı gırdıler ve cıktılar olarak ıkıye ayırdık
dataset_x = dataset[:, :12]
dataset_y = dataset[:, 12]

#Daha sonra datasetımıze hangı scaling uygulayacagımızı secmek ıcın, dataset ıcerısınde bulunan gırdı kolonlarının hıstogramını cızdırdık ve kolonların normal dagılım gostermedıgını gorduk ve bu durumda mms kullanacagız.
import pandas as pd
df=pd.DataFrame(dataset_x)
df.hist(figsize=(10,5))
import matplotlib.pyplot as plt
plt.show()

#Daha sonra datasetımızı train ve test olacak sekılde ayırdık.
from sklearn.model_selection import train_test_split
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

#Daha sonra datasetımıze scaling ıslemı uygulayacagız, cunku dataset ıcerısındekı sayılar bırbırınden cok fazla farklı degerlerde.
#Datasetımıze ohe uygulamadık, aslında ıkı farklı kolonda kategorık alanlar mevcut, fakat kategorı sayısı 2 oldugu ıcın bu kolonlara ohe donusumu uygulamaya gerek yoktur.
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
mms.fit(training_dataset_x)
training_dataset_x = mms.transform(training_dataset_x)


#Daha sonra model olusturup, modelımıze katmanlar ekledık
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential(name='Heart_Attack')
model.add(Dense(input_dim=dataset_x.shape[1], units=100, activation='relu', name='Hidden-1'))
model.add(Dense(units=100, activation='relu', name='Hidden-2'))
model.add(Dense(units=1, activation='sigmoid', name='Output'))


#Daha sonra modelımızı compile edıp fit ettık
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(x=training_dataset_x, y=training_dataset_y, batch_size=32,epochs=10, validation_split=0.2)

#en uygun epoch degerını ayarlamamız ıcın gereklı grafıklerın cızımını yaptıgımız kısım, bu grafıkten hareketle bu model ıcın en uygun epoch degerının 50 oldugunu gorduk
import matplotlib.pyplot as plt

figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Loss - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, len(hist.history['loss']) + 1), hist.history['loss'])
plt.plot(range(1, len(hist.history['val_loss']) + 1), hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Binary Accuracy - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Binary Accuracy')
plt.plot(range(1, len(hist.history['binary_accuracy']) + 1), hist.history['binary_accuracy'])
plt.plot(range(1, len(hist.history['val_binary_accuracy']) + 1), hist.history['val_binary_accuracy'])
plt.legend(['Binary Accuracy', 'Validation Binary Accuracy'])
plt.show()


#Daha sonra test datasetımızı de scale ettık ve daha sonra test datasetımız aracılıgı ıle evaluate ıslemınde bulunduk
test_dataset_x = mms.transform(test_dataset_x)
test_result = model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(test_result)):
    print(f'{model.metrics_names[i]}, {test_result[i]}')


#Daha sonra modelımızde kullancagımız predıct datalara da scale ıslemı yaptık.
predict_data = np.loadtxt('predicted_heart_failure_data.csv', delimiter=',')
# print(predict_data)
mms.fit(predict_data)
scaled_data = mms.transform(predict_data)

#Daha sonra da predict ıslemınde bulunduk
predicted_result = model.predict(scaled_data)

for i in range(len(predicted_result)):
    if predicted_result[i, 0] > 0.5:
        print('Bu kisi kalp krizinden ÖLEBİLİR')
    else:
        print('Bu kisi kalp krizinden ÖLMEZ')