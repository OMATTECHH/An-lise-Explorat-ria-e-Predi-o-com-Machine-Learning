import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 📌 Carregar os dados
df = pd.read_csv("dados.csv")

# 📊 Visualizar as primeiras linhas do dataset
print(df.head())

# 🔍 Estatísticas básicas
print(df.describe())

# 📈 Gráfico de distribuição da variável alvo
plt.figure(figsize=(8, 5))
sns.histplot(df["target"], bins=30, kde=True, color="blue")
plt.title("Distribuição da Variável Alvo")
plt.xlabel("Valor")
plt.ylabel("Frequência")
plt.show()

# 📊 Matriz de Correlação
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlação entre Variáveis")
plt.show()

# 🎯 Dividir dados em treino e teste
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Treinar Modelo de Machine Learning
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 📈 Fazer previsões
y_pred = modelo.predict(X_test)

# 📊 Comparação entre valores reais e previstos
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, color="red")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="blue", linestyle="--")
plt.xlabel("Valores Reais")
plt.ylabel("Previsões")
plt.title("Comparação: Valores Reais vs Previstos")
plt.show()

# 📊 Avaliação do Modelo
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Erro Absoluto Médio (MAE): {mae:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")