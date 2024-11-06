# FineTuning

Este repositório implementa uma rede neural com técnicas de **fine-tuning** para otimização do desempenho. Abaixo estão as etapas do processo, desde a criação da base de dados até a avaliação do modelo.

## 1. Base de Dados

Utilizamos uma base de dados sintética criada com a função `make_classification` da biblioteca `sklearn`. Essa função gera um conjunto de dados de classificação binária, ideal para testar e treinar algoritmos de aprendizado de máquina.

- **n_samples=1000**: Gera 1000 amostras.
- **n_features=10**: Cada amostra tem 10 características (ou variáveis).
- **n_classes=2**: Tarefa de classificação binária (0 ou 1).
- **n_informative=5**: Das 10 características, 5 são úteis para a classificação; as outras adicionam ruído.

## 2. Dividindo a Base em Treinamento e Teste

Dividimos os dados para treino e teste para avaliar o desempenho do modelo em dados novos.

- **test_size=0.2**: 20% dos dados são usados para teste, e 80% para treino.
- **random_state=42**: Define uma semente para que a divisão seja sempre a mesma, garantindo reprodutibilidade.

## 3. Normalização dos Dados

A normalização ajuda o modelo a convergir mais rápido e a lidar melhor com escalas diferentes nos dados. Utilizamos `StandardScaler` para transformar os dados para uma distribuição normal com média 0 e desvio padrão 1.

- `scaler.fit_transform(X_train)`: Calcula média e desvio padrão dos dados de treino e aplica a transformação.
- `scaler.transform(X_test)`: Aplica a mesma transformação nos dados de teste, mantendo a escala.

## 4. Construindo a Rede Neural

Criamos um modelo simples com `TensorFlow` e `Keras`, com três camadas ocultas e uma camada de saída.

- **Camada 1**: 64 neurônios com ativação `relu`. `input_shape` define a entrada com 10 características.
- **Camada 2**: 32 neurônios com ativação `relu`.
- **Camada 3**: 16 neurônios com ativação `relu`.
- **Camada de Saída**: 1 neurônio com ativação `sigmoid`, ideal para classificação binária (retorna valores entre 0 e 1).

## 5. Compilação do Modelo

O modelo é compilado com o otimizador **Adam** usando uma taxa de aprendizado inicial de 0,01 e a função de perda `binary_crossentropy`, adequada para problemas de **classificação binária**.

## 6. Aplicação do Fine-Tuning

Utilizamos duas técnicas principais de fine-tuning:

1. **Redução da Taxa de Aprendizado (`ReduceLROnPlateau`)**:  
   Este callback reduz a taxa de aprendizado em 50% (fator 0.5) se o modelo não melhorar o `val_loss` por 3 épocas consecutivas, ajustando os parâmetros de forma mais refinada nas últimas épocas.
   
   Importante destacar que a aplicação da redução dessa taxa deve ser feita apenas depois de algumas execuções de epochs, pois caso seja feita desde o inicio, **causaria um underfitting na solução**.

3. **Early Stopping**:  
   Interrompe o treinamento se o `val_loss` não melhorar por 5 épocas, restaurando os melhores pesos encontrados. Isso evita que o modelo continue treinando excessivamente e se ajuste ao ruído nos dados de treino.

## 7. Treinamento do Modelo com Fine-Tuning

O modelo é treinado com `model.fit`, aplicando os callbacks de *fine-tuning* para controlar a taxa de aprendizado e interromper o treinamento quando necessário.

## 8. Avaliação e Visualização dos Resultados

Após o treinamento, o modelo é avaliado nos dados de teste. Também plotamos a acurácia e a perda para monitorar a evolução do treinamento. Esses gráficos ajudam a visualizar como o fine-tuning influenciou o desempenho do modelo ao longo das épocas.

---

Esse processo demonstra o impacto positivo do fine-tuning no desempenho da rede neural, aprimorando a precisão e a generalização para novos dados.
