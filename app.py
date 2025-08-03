import gradio as gr
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
iris = load_iris()
x = iris.data
y = iris.target

# Preprocessing
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split into training and testing
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

# Evaluation
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Show results in console
print(f"Model accuracy: {accuracy:.2f}")
print("Confusion matrix: ")
print(conf_matrix)

# Interface
def predict_flower(sep_len, sep_wid, pet_len, pet_wid):
    input = np.array([[sep_len, sep_wid, pet_len, pet_wid]])
    input_scaled = scaler.transform(input)
    proba = model.predict_proba(input_scaled)[0]
    predicted_class = np.argmax(proba)
    class_name = iris.target_names[predicted_class]
    probability = proba[predicted_class]

    details ="\n".join(f"{iris.target_names[i]}: {p: .2%}" for i, p in enumerate(proba))
    return f"Predicted type: {class_name} ({probability: .2%})\n\nProbabilities:\n{details}"

interface = gr.Interface(fn=predict_flower, inputs=[gr.Number(label="Sepal lenght (cm)"), gr.Number(label="Sepal width (cm)"), gr.Number(label="Petal lenght (cm)"), gr.Number(label="Petal width (cm)")], outputs=gr.Text(label="Result"), title="Flowers classificator- kNN", description="Enter the measurements of the flower and predict its type.")
interface.launch()


