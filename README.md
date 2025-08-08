# 🌿 Detecção de Doenças em Plantas com IA: Comparativo entre CNN e Modelos Clássicos
_Plant Disease Detection with AI: A CNN vs. Classical ML Comparison_

**Autores:** [Rilson Joás](https://github.com/rilsonjoas) e [Ryan Eskinazi](https://github.com/reskyz1)

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-blueviolet.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📜 Visão Geral do Projeto

Este projeto explora a aplicação de técnicas de Inteligência Artificial e Visão Computacional para a detecção e classificação de doenças em folhas de plantas. Utilizando o dataset *New Plant Diseases Dataset (Augmented)*, que contém mais de 87.000 imagens de 38 classes diferentes, o objetivo principal é desenvolver um modelo preciso para identificar se uma planta está saudável ou qual doença a afeta.

O grande diferencial deste trabalho é a **análise comparativa** entre uma abordagem moderna de Deep Learning, utilizando uma **Rede Neural Convolucional (CNN)**, e algoritmos clássicos de Machine Learning, como **KNN, Árvore de Decisão e Naive Bayes**.

## ✨ Principais Características

- **Projeto de Ponta a Ponta:** Cobre desde o pré-processamento de imagens e extração de características até o treinamento, avaliação e análise comparativa dos modelos.
- **Extração de Características Clássicas:** Implementação de técnicas de visão computacional para extrair características de **cor (Histograma)** e **textura (GLCM)**.
- **Modelo de Deep Learning (CNN):** Construção e treinamento de uma Rede Neural Convolucional utilizando TensorFlow/Keras, com técnicas de **Data Augmentation** para robustez.
- **Modelos de Machine Learning Clássicos:** Treinamento e avaliação de modelos como K-Nearest Neighbors (KNN), Árvore de Decisão e Gaussian Naive Bayes com validação cruzada robusta (5x10).
- **Análise Comparativa Detalhada:** Avaliação aprofundada dos modelos com base em métricas como Acurácia, F1-Score, Matriz de Confusão e tempo de execução.

## 📊 Resultados e Análise

A análise comparativa demonstrou uma superioridade clara do modelo de Deep Learning para esta tarefa complexa de classificação de imagens.

### Desempenho Final dos Modelos

| Modelo                          | Dataset Utilizado              | Acurácia Média | F1-Score Ponderado Médio |
| ------------------------------- | ------------------------------ | :------------: | :----------------------: |
| **CNN**                         | Imagens (Augmented)            |   **95.13%**   |        **95.00%**        |
| **KNN (k=3)**                   | Bruto (Cor)                    |     88.52%     |          88.39%          |
| **Árvore de Decisão**           | Processado (Cor+Textura)       |     86.08%     |          85.99%          |
| **Naive Bayes**                 | Processado (Cor+Textura)       |     55.42%     |          53.32%          |

A **CNN alcançou uma acurácia de 95.13%**, superando significativamente o melhor modelo clássico, o **KNN com k=3 (88.52%)**, que obteve seu melhor resultado utilizando apenas características de cor. Curiosamente, a adição de características de textura melhorou o desempenho da Árvore de Decisão e do Naive Bayes, mas piorou o do KNN, indicando que a extração manual de características nem sempre generaliza bem para todos os algoritmos.

## 🛠️ Metodologia

O projeto foi dividido em quatro partes principais:

1.  **Extração de Características (Para Modelos Clássicos):**
    - As imagens foram processadas para gerar dois datasets em formato `.csv`:
        1.  **Base "Bruta"**: Contendo apenas histogramas de cor (512 características).
        2.  **Base "Pré-Processada"**: Contendo características de cor e textura (GLCM), totalizando 517 características.

2.  **Treinamento da CNN (Deep Learning):**
    - Uma CNN foi construída do zero com TensorFlow/Keras.
    - O modelo utiliza camadas de Convolução, Batch Normalization, MaxPooling e Dropout para aprender características diretamente das imagens.
    - `ImageDataGenerator` foi usado para aplicar *Data Augmentation* em tempo real, aumentando a diversidade dos dados de treino.

3.  **Treinamento dos Modelos Clássicos:**
    - Os algoritmos KNN, Árvore de Decisão e Naive Bayes foram treinados com os datasets `.csv`.
    - Foi aplicado um esquema de validação cruzada **5x10 `RepeatedKFold`** para garantir uma avaliação estatisticamente robusta do desempenho.

4.  **Análise Final:**
    - Os resultados de todos os modelos foram agregados, e gráficos comparativos foram gerados para analisar a acurácia, o F1-score e o tempo de execução.

## 🚀 Como Executar o Projeto

**1. Pré-requisitos:**
   - Python 3.10 ou superior
   - Um ambiente virtual (recomendado: `venv` ou `conda`)

**2. Clone o Repositório:**
   ```bash
   git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
   cd SEU-REPOSITORIO
   ```

**3. Crie e Ative um Ambiente Virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

**4. Instale as Dependências:**
   - Crie um arquivo `requirements.txt` a partir do seu ambiente e adicione-o ao repositório. O comando para criar é:
   ```bash
   pip freeze > requirements.txt
   ```
   - Para instalar:
   ```bash
   pip install -r requirements.txt
   ```

**5. Estrutura de Pastas e Dataset:**
   - Faça o download do [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) do Kaggle.
   - Crie uma pasta `datasets/` na raiz do projeto.
   - Descompacte o dataset de forma que a estrutura final seja:
     ```
     datasets/
     └── New Plant Diseases Dataset(Augmented)/
         └── New Plant Diseases Dataset(Augmented)/
             ├── train/
             └── valid/
     ```

**6. Execute o Notebook:**
   - Inicie o Jupyter Notebook ou JupyterLab:
   ```bash
   jupyter notebook
   ```
   - Abra o arquivo `projeto_ia.ipynb` e execute as células em ordem.

## 💻 Tecnologias Utilizadas

- **Python 3.10**
- **TensorFlow & Keras:** Para a construção e treinamento da CNN.
- **Scikit-learn:** Para os modelos clássicos de Machine Learning e métricas de avaliação.
- **OpenCV & Scikit-image:** Para processamento de imagem e extração de características.
- **Pandas & NumPy:** Para manipulação de dados.
- **Matplotlib & Seaborn:** Para visualização dos resultados.

---
**Observação:** O notebook foi executado em um ambiente com GPU para acelerar o treinamento da CNN. O treinamento na CPU é possível, mas significativamente mais lento.
