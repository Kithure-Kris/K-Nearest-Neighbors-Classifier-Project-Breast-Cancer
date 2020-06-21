import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)
print(breast_cancer_data.target) #gives you the labels of every data point.
print(breast_cancer_data.target_names) #By looking at the target_names, we know that 0 corresponds to malignant.

#Splitting the data into Training and Validation Sets
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)
print(len(training_data)) #455
print(len(training_labels)) #455

#Running the Classifier
accuracies = []
for n_neighbors in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors)
  classifier.fit(training_data, training_labels)
  #How accurate is the validation set?
  score = classifier.score(validation_data, validation_labels)
  accuracies.append(score)
  #print(f'When k is {n_neighbors} = {score}')
#Best accuracy when k = 23/24 ;0.964912

#Plotting
k_list = range(1, 101)
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()