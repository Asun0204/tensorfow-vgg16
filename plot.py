import pickle
import matplotlib.pyplot as plt

with open('temp.pickle', 'rb') as file:
    data = pickle.load(file)

losses = data['loss']
accs = data['accuracy']

plt.figure()
sub1 = plt.subplot(211)
sub1.set_title('loss')
plt.plot(losses)
sub2 = plt.subplot(212)
sub2.set_title('accuracy')
plt.plot(accs)
plt.show()