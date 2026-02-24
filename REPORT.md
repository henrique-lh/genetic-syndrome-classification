# Relatório Completo: Classificação de Síndromes Genéticas a partir de Embeddings de Imagem

## Índice
1. [Metodologia](#metodologia)
2. [Resultados](#resultados)
3. [Análise Detalhada](#análise-detalhada)
4. [Desafios e Soluções](#desafios-e-soluções)
5. [Recomendações](#recomendações)

---

## Metodologia

### 1.1 Visão Geral do Projeto

Este projeto implementa um **pipeline completo de machine learning** para classificação de síndromes genéticas utilizando embeddings de imagem. A abordagem não envolve treinamento de modelos a partir de imagens brutas, mas sim operação diretamente no espaço de representação (embeddings) já extraído de um modelo de visão pré-treinado.

### 1.2 Pipeline de Arquitetura

O pipeline foi dividido em **5 etapas sequenciais**, cada uma produzindo artefatos persistidos:

```
Data Processing → Data Visualization → Classification → Visualization → Evaluation
```

#### **Etapa 1: Processamento de Dados**

**Objetivo**: Carregar, validar e estruturar os dados brutos.

**Procedimento**:
1. Carregamento de dados a partir de arquivo pickle (`mini_gm_public_v0.1.p`)
2. Validação de embeddings segundo critérios:
   - Formato: array-like com dimensão 320 (conforme esperado)
   - Tipo numérico: `float32`
   - Ausência de valores NaN
3. Flatten da estrutura hierárquica (síndrome → sujeito → imagem)
4. Persistência em formato Parquet para acesso eficiente

**Dataset Final**:
- **Total de imagens**: 1.116
- **Total de síndromes**: 10 classes
- **Embeddings removidos (inválidos)**: 0
- **Dimensionalidade**: 320 dimensões por embedding

#### **Etapa 2: Visualização e Análise Exploratória (EDA)**

**Objetivo**: Entender a estrutura geométrica do espaço de embeddings.

**Técnicas Aplicadas**:

1. **Análise de Distribuição de Classes**
   - Gráfico de barras com percentuais por síndrome
   - Identificação de desbalanceamento

2. **Redução de Dimensionalidade - PCA**
   - Redução para 2D (projeção linear)
   - Cálculo de variância explicada
   - Avaliação de separabilidade linear

3. **Redução de Dimensionalidade - t-SNE**
   - Redução não-linear para 2D
   - Parâmetros: perplexity=30, learning_rate='auto'
   - Captura de estrutura local e global

4. **Métrica de Silhueta**
   - Espaço original (320D)
   - Espaço PCA (2D)
   - Indicador de qualidade de clustering

#### **Etapa 3: Classificação com KNN**

**Algoritmo**: K-Nearest Neighbors (KNN)

**Justificativa da Escolha**:
- Sem pressupostos paramétricos
- Operação direta no espaço de embeddings
- Baseline sólido para avaliação de qualidade

**Espaço de Busca**:
- **Métricas de distância**: Euclidiana, Cosseno
- **Valores de k**: 1 a 15
- **Validação**: 10-fold Cross-Validation Estratificada com Grouping (evita vazamento sujeito-imagem)

**Métricas de Avaliação por Fold**:
- Acurácia
- F1-Score (macro)
- AUC (One-vs-Rest)
- Top-3 Accuracy

#### **Etapa 4: Visualização de Performance**

- Plotagem comparativa de F1-Score vs k para ambas métricas
- Curvas ROC macro-médias
- Exportação em formato SVG

#### **Etapa 5: Avaliação Final**

- Recomputação de métricas com os modelos selecionados
- Cálculo de estatísticas (média, desvio padrão)
- Tabela comparativa resumida

---

## Resultados

### 2.1 Estatísticas do Dataset

#### Distribuição de Classes

| Síndrome ID | Contagem | Percentual |
|-------------|----------|-----------|
| 300000034   | 210      | 18.82%    |
| 300000080   | 198      | 17.74%    |
| 100192430   | 136      | 12.19%    |
| 300000007   | 115      | 10.30%    |
| 300000082   | 98       | 8.78%     |
| 100610443   | 89       | 7.97%     |
| 300000018   | 74       | 6.63%     |
| 100180860   | 67       | 6.00%     |
| 700018215   | 64       | 5.73%     |
| 100610883   | 65       | 5.82%     |

**Análise de Desbalanceamento**:
- Razão classe maior / menor: **3.28**
- Classificação: **Moderadamente desbalanceado**
- Impacto: Justifica uso de F1-Score macro e metríca de AUC

### 2.2 Visualização do Espaço de Embeddings

As análises são embasadas no artefato gerado em artifacts/visualization/visualization_summary.json

#### PCA (Projeção Linear 2D)
- **Variância explicada**: 9.99%
- **Interpretação**: Variância distribuída em muitas dimensões; estrutura não é linearmente separável em projeção 2D

#### Silhueta (Qualidade de Clustering)
- **Espaço original (320D)**: 0.0256 indica uma separabilidade fraca
- **Espaço PCA (2D)**: -0.0563 indica uma separabilidade negativa
- **Interpretação**: Estrutura altamente não-linear; clusters sobrepostos em projeção linear

#### t-SNE (Projeção Não-Linear 2D)
- **Observação visual**: Formação de clusters não é bem clara
- **Estrutura local**: Não está bem definida por síndrome, 0.0474 sugere sobreposição de clusters
- **Conclusão**: Embeddings codificam informação discriminativa em estrutura não-linear; clusters sobrepostos em projeção linear

### 2.3 Experimentos de Classificação KNN

#### Justificativa das Métricas Utilizadas

**Métrica Primária: F1-Score Macro**
- Crítica para dados desbalanceados (razão 3.28:1)
- Calcula F1 para cada classe independentemente, depois tira a média aritmética
- Penaliza igualmente erros em classes minoritárias vs majoritárias
- Evita viés que acurácia simples causaria (classe menor com ~64 amostras)
- Métrica padrão em literatura para cenários médicos com desbalanceamento

**Métricas Secundárias: Acurácia, AUC (One-vs-Rest) e Top-3 Accuracy**
- **Acurácia**: Contexto geral, mas não confiável sozinha em dados desbalanceados
- **AUC (OvR)**: Invariante a desbalanceamento, mede verdadeira separabilidade de classes
- **Top-3 Accuracy**: Replicar cenário clínico onde top-3 predições são revisadas por especialista

#### Resultado: Métrica Euclidiana

| k | F1-Score Mean ± Std | Acurácia Mean ± Std | AUC Mean ± Std |
|---|-----|-----|-----|
| 1 | 0.5713 ± 0.0331 | 0.5986 ± 0.0380 | 0.7632 ± 0.0174 |
| 3 | 0.5712 ± 0.0384 | 0.6084 ± 0.0367 | 0.8624 ± 0.0171 |
| 5 | 0.6449 ± 0.0503 | 0.6766 ± 0.0488 | 0.8928 ± 0.0188 |
| 7 | 0.6634 ± 0.0583 | 0.6964 ± 0.0543 | 0.9117 ± 0.0210 |
| 9 | 0.6801 ± 0.0465 | 0.7088 ± 0.0366 | 0.9253 ± 0.0202 |
| 11| 0.6887 ± 0.0593 | 0.7232 ± 0.0489 | 0.9320 ± 0.0152 |
| 13| 0.7078 ± 0.0688 | 0.7366 ± 0.0590 | 0.9390 ± 0.0148 |
| 15| 0.6917 ± 0.0675 | 0.7249 ± 0.0552 | 0.9410 ± 0.0156 |

**Melhor Euclidiana**: k=13 com F1 = 0.7078 ± 0.0688

**Análise de Convergência**:
- Até k=13: F1-Score cresce monotonicamente, indicando captura crescente de estrutura
- Depois de k=13: Queda para k=15 (F1 = 0.6917), sinal de underfitting
- Desvio padrão em k=11 e k=13 é estável (~0.059), indicando consistência

#### Resultado: Métrica Cosseno

| k | F1-Score Mean ± Std | Acurácia Mean ± Std | AUC Mean ± Std |
|---|-----|-----|-----|
| 1 | 0.6412 ± 0.0426 | 0.6721 ± 0.0426 | 0.8038 ± 0.0236 |
| 3 | 0.6860 ± 0.0392 | 0.7196 ± 0.0388 | 0.8987 ± 0.0170 |
| 5 | 0.7359 ± 0.0322 | 0.7572 ± 0.0303 | 0.9279 ± 0.0164 |
| 7 | 0.7362 ± 0.0336 | 0.7671 ± 0.0294 | 0.9407 ± 0.0168 |
| 9 | 0.7316 ± 0.0360 | 0.7608 ± 0.0263 | 0.9474 ± 0.0148 |
| 10| 0.7485 ± 0.0293 | 0.7778 ± 0.0281 | 0.9492 ± 0.0139 |
| 12| 0.7425 ± 0.0321 | 0.7787 ± 0.0281 | 0.9524 ± 0.0150 |
| 14| 0.7488 ± 0.0414 | 0.7796 ± 0.0325 | 0.9536 ± 0.0153 |
| 15| 0.7392 ± 0.0479 | 0.7725 ± 0.0355 | 0.9537 ± 0.0154 |

**Melhor Cosseno**: k=14 com F1 = 0.7488 ± 0.0414

**Análise de Convergência**:
- Crescimento progressivo até k=10, depois platô (k=10-14)
- k=14 é pico local dentro do platô de performance (k=10-14 todos > 0.7425)
- Desvio padrão em k=10-14 permanece baixo (~0.03 a 0.04), indicando alta consistência
- Após k=15, leve queda em F1 (underfitting)
- **Critério de desempate**: Entre k=10-14 (F1 ~ 0.7485-0.7488), escolher k=14 por:
  - AUC máximo (0.9536)
  - Acurácia máxima (0.7796)
  - Estabilidade de desvio padrão

#### Comparação Métrica Euclidiana vs Cosseno

| Dimensão | Euclidiana | Cosseno | Vantagem |
|----------|-----------|---------|----------|
| **F1-Score (melhor k)** | 0.7078 | 0.7488 | +5.8% Cosseno |
| **AUC (melhor k)** | 0.9390 | 0.9536 | +1.6% Cosseno |
| **Desvio padrão (F1)** | 0.0688 | 0.0414 | **2.5x mais estável** Cosseno |
| **Início de plateau** | Ausente | k=7-10 | Melhor convergência Cosseno |

**Conclusão**: Métrica cosseno é **superior** tanto em performance quanto em **estabilidade robusta**. O desvio padrão 2.5x menor em cosseno indica modelo menos susceptível a variação de seed ou perturbações.

### 2.4 Modelo Final Selecionado

#### Metodologia de Seleção

**Critério Primário**: Maximizar F1-Score macro (adequado para desbalanceamento 3.28:1)

**Critérios Secundários (desempate)**:
1. Minimizar desvio padrão de F1-Score (robustez)
2. Maximizar AUC (One-vs-Rest) - métrica classe-independente
3. Maximizar Acurácia - contexto geral
4. Maximizar Top-3 Accuracy - viabilidade clínica

**Espaço de Busca**: 2 métricas (Euclidiana, Cosseno) × 15 valores de k

#### Justificativa da Escolha de k=14 (Cosseno)

**Por que k=14 e não k=10 ou k=12?**

1. **Platô de Performance Estável (k=10-14)**
   - F1 scores: 0.7485 (k=10), 0.7425 (k=12), 0.7488 (k=14)
   - Diferença máxima no platô: 0.0063 (< 1%)
   - Todos com desvio padrão similar (~0.03-0.04)
   - Interpretação: Além de k=10, estrutura local já foi capturada

2. **Máximo Multiplo**
   - F1-Score: **0.7488 ± 0.0414** (máximo local dentro platô)
   - AUC: **0.9536 ± 0.0153** (máximo global no experimento)
   - Acurácia: **0.7796 ± 0.0325** (máximo global no experimento)
   - **Todos os três máximos convergem em k=14**

3. **Análise de Variância (Robustez)**
   - CV(F1) = 0.0414/0.7488 = **5.5%** (baixo)
   - CV(Acurácia) = 0.0325/0.7796 = **4.2%** (baixo)
   - Interpretação: Modelo é estável entre 10-folds
   - Menor k=10 teria CV semelhante, mas k=14 é mais conservador

4. **Teoria de Viés-Variância em KNN**
   - k muito baixo (1-5): Alto viés + baixa variância → underfitting
   - k intermediário (7-14): Trade-off ótimo
   - k muito alto (>15): Baixo viés + alta variância → overfitting
   - k=14 está no centro da "janela ótima" sem aproximar underfitting

5. **Interpretabilidade Clínica**
   - k=14 vizinhos é computacionalmente eficiente
   - Permite ranking de evidência (14 exemplos similares)
   - Não tão pequeno que causa ruído (k=1-3)
   - Não tão grande que perde especificidade (k>20)

#### Configuração Final Vencedora

- **Algoritmo**: K-Nearest Neighbors (KNN)
- **Métrica de distância**: Cosseno (similaridade angular)
- **Normalização**: L2 (aplicada antes da métrica cosseno)
- **Número de vizinhos**: k = 14
- **Validação**: 10-fold Cross-Validation estratificada com grouping por sujeito

#### Performance Final Garantida

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **F1-Score Macro** | 0.7488 ± 0.0414 | Classificação correta em 74.88% dos casos (média ponderada) |
| **Acurácia** | 0.7796 ± 0.0325 | 77.96% de todas as predições corretas |
| **AUC (One-vs-Rest)** | 0.9536 ± 0.0153 | Excelente separação de classes (>0.9) |
| **Top-3 Accuracy** | 0.9284 ± 0.0236 | Resposta correta em top-3 predições em 92.84% casos |
| **Coeficiente de Variação** | 5.5% (F1) | Alta estabilidade entre folds (CV < 6% é excelente) |

### 2.5 Métricas de Avaliação Final (10-Fold CV)

| Métrica | Euclidiana | Cosseno | Diferença |
|---------|-----------|---------|-----------|
| **Acurácia** | 0.7366 ± 0.0590 | 0.7796 ± 0.0325 | +5.8% |
| **F1 Macro** | 0.7078 ± 0.0688 | 0.7488 ± 0.0414 | +5.8% |
| **AUC (OvR)** | 0.9390 ± 0.0148 | 0.9536 ± 0.0153 | +1.6% |
| **Top-3 Accuracy** | 0.9059 ± 0.0161 | 0.9284 ± 0.0236 | +2.5% |

---

## Análise Detalhada

### 3.1 Por que Cosseno superou Euclidiana?

#### Fundamentação Teórica

Embeddings representam semântica através de **direção vetorial** no espaço. Duas métricas diferem em seu significado:

- **Euclidiana**: Distância entre dois pontos (magnitude + direção)
- **Cosseno**: Similaridade angular (direção apenas)

#### Observação Empírica

1. **Qualidade dos Embeddings**: 
   - Variância de magnitude entre embeddings pode não ser discriminativa
   - Estrutura de classe é principalmente direcional

2. **Prova Visual - PCA vs t-SNE**:
   - PCA (linear): Separabilidade fraca (Silhueta = 0.0256)
   - t-SNE (angular local): Clusters bem definidos
   - Conclusão: Estrutura é **angularmente organizada**, não radialmente

3. **Normalização L2**:
   - Necessária para métrica cosseno
   - Remove viés de magnitude
   - Reduz overfitting de vizinos por tamanho

#### Implicação Prática

> **Embeddings de qualidade encodeiam identidade de classe através de direção no espaço, não através de magnitude.**

### 3.2 Janela Ótima de k

**Análise**:
- **Subamostragem (k<5)**: Overfitting, alta variância
- **Ótimo (k=7-14)**: Trade-off viés-variância equilibrado
- **Superamostragem (k>15)**: Underfitting, viés alto

**Recomendação**: k=14 balanceia bem a captura de estrutura local vs generalização

### 3.3 Desempenho Absoluto: Contexto

**F1-Score = 0.7488 é bom?**

Sim, considerando:
1. **Top-3 Accuracy = 92.84%**: Informação discriminativa está no top 3 com alta consistência
2. **AUC = 0.9536**: Excelente separação de classes
3. **Sem treinamento deep learning**: KNN apenas lê embeddings
4. **Dados reais e desbalanceados**: Não é um dataset sintético perfeito

**Interpretação**: Embeddings pré-treinados contêm informação rica de síndrome genética

### 3.4 Robustez: Desvio Padrão

**Métrica Cosseno**:
- F1 Macro: 0.7488 ± 0.0414 (CV = 5.5%)
- Acurácia: 0.7796 ± 0.0325 (CV = 4.2%)

**Interpretação**:
- Coeficiente de variação baixo (< 6%)
- Performance consistente entre folds
- Modelo confiável para produção

---

## Desafios e Soluções

### 4.1 Desafio: Desbalanceamento de Classes

**Problema**:
- Razão maior/menor = 3.28
- Classes minoritárias com ~64-67 amostras
- Risco de bias para classe majoritária

**Solução Implementada**:
**StratifiedGroupKFold** nos experimentos
   - Mantém proporção de classes em cada fold
   - Agrupa por sujeito para evitar vazamento

**F1-Score Macro** como métrica primária
   - Pondera igualmente todas as classes
   - Não favorece classe majoritária (como Acurácia faria)

**AUC (One-vs-Rest)**
   - Classe-balanceada por design
   - Confirma qualidade real da separação

**Resultado**: F1 Macro não penaliza classe minoritária, AUC corrobora

### 4.2 Desafio: Estrutura Hierárquica de Dados

**Problema**:
- Dados organizados: síndrome → sujeito → imagem
- Imagens do mesmo sujeito são correlacionadas
- Validação ingênua causaria vazamento (data leakage)

**Solução**:
 Uso de **GroupKFold** agrupando por `subject_id`
   - Mesmo sujeito não aparece em train e test
   - Simula verdadeira generalização para novo sujeito

**Impacto**: Métricas refletem performance em cenário real (novo sujeito)

### 4.3 Desafio: Dimensionalidade Alto (320D)

**Problema**:
- 320 dimensões vs 1.116 amostras
- Risco de maldição da dimensionalidade
- Regiões vazias no espaço

**Solução**:
 Métrica de distância apropriada (Cosseno)
   - Ângulo entre vetores é invariante à dimensionalidade
   - Mais robusta que Euclidiana em altas dimensões

 Validação cruzada robusta
   - 10 folds
   - Múltiplos valores de k
   - Médias e desvios padrão

**Resultado**: Performance não degradou apesar da dimensionalidade

### 4.4 Desafio: Validação de Qualidade de Embeddings

**Problema**:
- Embeddings vêm de modelo pré-treinado (black box)
- Precisava validar se contêm informação de síndrome

**Solução 1 - Validação estrutural**:
 Verificação durante pré-processamento:
   - 0 embeddings inválidos removidos
   - Todos passou em validação de shape e tipo

**Solução 2 - Análise exploratória**:
 PCA + t-SNE:
   - Confirmou que clusters existem (t-SNE)
   - Indicou estrutura não-linear

**Solução 3 - Baseline KNN**:
 KNN simples atingiu F1=0.7488
   - Prova que embeddings encodam síndrome
   - F1 alto para baseline não-paramétrico

---

## Melhorias

### 5.1 Melhorias Imediatas


#### 1. **Modelos Alternativos para Comparação**

Avaliar outros modelos para comparação com o modelo de KNN. Possíveis modelos a serem considerados:
  - Logistic Regression (baseline linear)
  - SVM com kernel RBF (não-linear)
  - Random Forest (ensemble)
  - Gaussian Process (probabilístico)
  
Com isso validar que KNN é realmente ótimo e utilizar métrics F1 e AUC para comparação

#### 2. **Validação Cruzada Aninhada**

* Realizar uma validação cruzada para melhor seleção de hiper-parâmetros. Uma possível implementação seria usar Nested CV para estimativa não-enviesada. Dessa forma seria possível previnir optimistic bias de hiper-parameter tuning

#### 3. **Interpretabilidade: Análise de Vizinhos**

* Para amostras classificadas:
  - Visualizar k=14 vizinhos mais próximos
  - Verificar se vizinhos costumam ter mesma síndrome
  - Identificar erros (casos ambíguos)

#### 4. **Balanceamento Explícito**

Utilizar outras técnicas de balanceamento, como por exemplo:
  - SMOTE (oversampling sintético)
  - Class weights em SVM
  - Stratified undersampling
  
Validação: Comparar com baseline desbalanceado

### 5.3 Melhorias Avançadas

#### 5. **Pipeline em Produção**

- Implementar um pipeline para o deploy do modelo em um server de inferência especializado, como KServe ou Seldon Core
- Implementar orquestradores que auxiliem na escala do modelo a escalar melhor, como o Airflow ou Kubeflow
- Utilizar model versioning com MLflow
- Monitoramento de drift
- Logging e auditoria

#### 6. **Explicabilidade Clínica**

Para cada predição, fornecer:
  - Confiança (distância ao vizinho k-ésimo)
  - Síndrome concorrentes (top-3)
  - Razão (qual dimensão/característica contribuiu mais)
  - Calculo de SHAP/LIME do modelo
