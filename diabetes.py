import numpy as np

#Datasetımızı okuduk
dataset = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
print(dataset)



#Daha sonra datasetımızı gırdıler ve cıktılar ıcın, egıtım-test asamasında kullanılma oranlarına gore bolduk.
from sklearn.model_selection import train_test_split
dataset_x = dataset[:, :8]
dataset_y = dataset[:, 8]
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

#Daha sonra datasetımızın hıstını cızdırdık kı, burada bır scaling  ıslemı yapacagız, sutundakı degerler normal olarak dagılmadıgı ıcın bız burada minmax scale kullancaz.
import pandas as pd
df=pd.DataFrame(dataset_x)
df.hist(figsize=(10,5))
import matplotlib.pyplot as plt
plt.show()

#Data setımızde sadece sayılar vardı, fakat sayıların degerlerı arasında fark vardı, bu yuzden de minmax scalıng kullandık.
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
mms.fit(training_dataset_x)
training_dataset_x = mms.transform(training_dataset_x)

#Daha sonra model yarattık ve modelımıze 2 hıdden, 1 gırdı, 1 de cıktı katmanını ekledık.
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(input_dim=8, units=100, activation='relu', name='Hidden-1'))
model.add(Dense(units=100, activation='relu', name='Hidden-2'))
model.add(Dense(units=1, activation='sigmoid', name='Output'))


#modelımızı compile ettık, daha sonra fit ederek modelı egıttık
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(x=training_dataset_x, y=training_dataset_y, batch_size=32, epochs=50, validation_split=0.2)

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

#Test datasetımızı de maxmin scale ıslemıne soktuktan sonra modelımızı teste tabii tuttuk. Test sonucunda elde ettıgımız loss ve binary_accuracy degerlerını de ekrana bastık
test_dataset_x = mms.transform(test_dataset_x)
test_result = model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(test_result)):
    print(f'{model.metrics_names[i]} ---> {test_result[i]}')


#Test ıslemınden sonra da predıct datamızı da scalinge tabii tuttuk ve scale edılmıs predıct datamıza predıct ıslemı yaptık ve bır kestırım sonucu elde ettık.
predict_data = np.array([[7,114,66,0,0,32.8,0.258,42]])
predict_data = mms.transform(predict_data)
predict_result = model.predict(predict_data)
if predict_result[0] > 0.5:
    print('Diyabetli')
else:
    print('Diyabetsiz')


#Daha sonra modelımızı save ettık, sonra da modelımızdekı mms nesnesını de save ettık
model.save('diabetes.h5')
import pickle
with open('diabetes.dat', 'wb') as f:
    pickle.dump(mms, f)
