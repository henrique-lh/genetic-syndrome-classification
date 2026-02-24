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

#### PCA (Projeção Linear 2D)
- **Variância explicada**: 9.99%
- **Interpretação**: Variância distribuída em muitas dimensões; estrutura não é linearmente separável em projeção 2D

#### Silhueta (Qualidade de Clustering)
- **Espaço original (320D)**: 0.0256 indica uma separabilidade fraca
- **Espaço PCA (2D)**: -0.0563 indica uma separabilidade negativa
- **Interpretação**: Estrutura altamente não-linear; clusters sobrepostos em projeção linear

#### t-SNE (Projeção Não-Linear 2D)
- **Observação visual**: Formação clara de clusters
- **Estrutura local**: Bem definida por síndrome
- **Conclusão**: Embeddings codificam informação discriminativa em estrutura não-linear

### 2.3 Experimentos de Classificação KNN

#### Métrica Euclidiana

| k | F1-Score |
|---|----------|
| 1 | 0.5713   |
| 3 | 0.5712   |
| 5 | 0.6449   |
| 7 | 0.6634   |
| 9 | 0.6801   |
| 11| 0.6887   |
| 13| 0.7078   |
| 15| 0.6917   |

**Melhor Euclidiana**: k=13, F1=0.7078

#### Métrica Cosseno

| k | F1-Score |
|---|----------|
| 1 | 0.6412   |
| 3 | 0.6860   |
| 5 | 0.7359   |
| 7 | 0.7362   |
| 9 | 0.7316   |
| 10| 0.7485   |
| 12| 0.7425   |
| 14| 0.7488   |
| 15| 0.7392   |

**Melhor Coseno**: k=14, F1=0.7488

**Melhoria Relativa**: (0.7488 - 0.7078) / 0.7078 = **5.8% de ganho**

### 2.4 Modelo Final Selecionado

**Critério de Seleção**: Máximo F1-Score macro durante cross-validation

**Configuração Vencedora**:
- **Tipo**: KNN
- **Métrica de distância**: Cosseno
- **Número de vizinhos (k)**: 14
- **Normalização**: L2 (usada na métrica cosseno)
- **F1-Score**: 0.7488 ± 0.0414
- **AUC (OvR)**: 0.9536 ± 0.0153

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

### 5.1 Melhorias Imediatas (Curto Prazo)

#### 1. **Divisão Rigorosa Train/Test**
```python
Recomendação: Implementar split temporal ou estratificado
Benefício: Avaliação final em dados verdadeiramente invisíveis
Impacto: Estimativa mais realista de erro de generalização
```

#### 2. **Modelos Alternativos para Comparação**
```python
Candidatos:
  - Logistic Regression (baseline linear)
  - SVM com kernel RBF (não-linear)
  - Random Forest (ensemble)
  - Gaussian Process (probabilístico)
  
Benefício: Validar que KNN é realmente ótimo
Métrica: Comparar F1, AUC e tempo
```

#### 3. **Validação Cruzada Aninhada**
```python
Atual: Cross-validation simples para hiper-parâmetros
Recomendado: Nested CV para estimativa não-enviesada
Razão: Previne optimistic bias de hiper-parameter tuning
```

#### 4. **Interpretabilidade: Análise de Vizinhos**
```python
Para amostras classificadas:
  - Visualizar k=14 vizinhos mais próximos
  - Verificar se vizinhos costumam ter mesma síndrome
  - Identificar erros (casos ambíguos)
  
Benefício: Entender por que modelo acerta/erra
```

### 5.2 Melhorias Médias (Médio Prazo)

#### 5. **Metric Learning e Distance Tuning**
```python
Explorar:
  - Mahalanobis distance (weighted Euclidean)
  - Siamese networks (aprender métrica)
  - Triplet loss (refinar embeddings)

Expectativa: F1 pode aumentar de 0.75 para 0.80+
```

#### 6. **Balanceamento Explícito**
```python
Técnicas:
  - SMOTE (oversampling sintético)
  - Class weights em SVM
  - Stratified undersampling
  
Validação: Comparar com baseline desbalanceado
```

#### 7. **Feature Engineering em Embeddings**
```python
Explorar transformações:
  - Estatísticas (mean, std, kurtosis por dimensão)
  - PCA reduzido (32-64D mantendo ~50% variância)
  - Clustering de embeddings (k-means features)
  
Benefício: Reduzir dimensionalidade, possível ganho de performance
```

### 5.3 Melhorias Avançadas (Longo Prazo)

#### 8. **Pipeline em Produção**
```python
Implementar:
  - Implementar um serve de inferência mais robusto (utilizando kserve, por exemplo)
  - Implementar orquestradores que auxiliem na escala do modelo a escalar melhor, como o Airflow ou Kubeflow Pipelines
  - Model versioning (MLflow)
  - Monitoramento de drift
  - Logging e auditoria
  
Objetivo: Deploy seguro em ambiente clínico
Considerações: Validação regulatória (se aplicável)
```

#### 9. **Explicabilidade Clínica**
```python
Para cada predição, fornecer:
  - Confiança (distância ao vizinho k-ésimo)
  - Síndrome concorrentes (top-3)
  - Razão (qual dimensão/característica contribuiu mais)
  
Benefício: Clinicamente interpretável, aumenta confiança
```

#### 10. **Ensembles e Stacking**
```python
Testar:
  - Votação entre KNN + Logistic Regression + SVM
  - Stacking com meta-learner
  - Bagging com diferentes valores de k
  
Benefício: Reduzir variância, possível melhoria de 1-2% em F1
```
