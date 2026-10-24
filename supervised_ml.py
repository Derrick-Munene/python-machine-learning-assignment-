from sklearn.linear_model import LinearRegression
#create a database
x = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
#label
y = [30,40,45,55,65,70,80,85,90,95]
#create a model
model = LinearRegression()
#Training a model
model.fit(x , y)
# prediction
hours = float(input("Enter the number of hours studied: "))
predicted_score = model.predict([[hours]])
print(f"predicted examination score: {predicted_score[0] :.2f}")

import matplotlib.pyplot as plt
#plot the original data points 
plt.scatter(x,y, color='blue', label='Actual Data')
#plot the model's prediction line
y_pred = model.predict(x)
plt.plot(x,y_pred, color='red' , label='Regression Line')
# Add labels and title 
plt.xlabel('Hours studied')
plt.ylabel('Examination Score')
plt.title('Hours Studied vs Exam Score')
plt.legend()
# show the plot
plt.show()
print(f"Model Formula: Score = {model.coef_[0]: .2f} *Hours + {model.intercept_:.2f}")
                       