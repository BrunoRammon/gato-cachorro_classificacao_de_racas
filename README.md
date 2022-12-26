# Análise de algoritmos de aprendizado supervisionado em um problema de classificação de raças de cães e gatos

Neste projeto apresentamos foi realizada uma solução de aprendizado de máquina envolvendo um problema de classificação. Em nosso caso, o problema envolve a predição de duas raças de gatos e três de cachorros a partir de imagens.

## Base de dados não tratados

A base de dados utilizada neste trabalho foi a The Oxford-IIIT Pet, que contém fotografias de cães e gatos distribuídos em 37 categorias com aproximadamente 200 imagens cada. No nosso caso utilizamos apenas 5 destas categorias, 2 para os felinos e 3 para cachorros, a saber: Siamês e Maine Coon de gatos; Saint Bernard, Newfoundland e German Shorthaired de cães. O dataset contendo todas as raças pode ser encontrado no site do grupo de pesquisa em computação visual da Universidade de Oxford[^1].

## Extração de características via HOG
Após a seleção manual das $1000$ fotografias que compõem as $5$ classes de animais que utilizamos, encontramos o nosso primeiro desafio, que foi o de transformar as imagens em dados inteligíveis para o computador. Desta forma, partimos para a parte de extração de características das referidas imagens, fazendo uso do _Histogram of Oriented Gradients_ (HOG). Geralmente, essa técnica é utilizada em visão computacional e processamento de imagens para de detecção de objetos. De forma simplificada, esse descritor calcula as frequências de orientação de gradientes ou direção das bordas, em porções de uma imagem. Ou seja, a imagem é dividida em pequenas regiões, chamadas de células, e para cada pixel dentro desta célula, há um histograma da orientação de gradiente. O descritor é a concatenação destes histogramas[^2].

Antes da aplicação do HOG, também realizamos a redefinição dos tamanhos das imagens, de modo a criarmos duas versões diferentes das imagens referentes a cada grupo. O tamanho das imagens redimensionadas foi de 128x128 pixels. Na aplicação do HOG para gerar a primeira base de dados, foi utilizado 16x16 pixels por célula. Na segunda base, por sua vez, o tamanho de uma célula foi de 20x20 pixels. Esse redimensionamento das imagens serviu para que os valores extraídos dos diferentes tamanhos fossem distintos, gerando assim maior variabilidade para testes futuros.

Logo em seguida, aplicamos o HOG a estas duas bases distintas, gerando, com isso, dois _datasets_: o primeiro com $1764$ atributos e o segundo com $900$ atributos. Em suma, os dois arquivos criados nada mais são do que tabelas no formato csv com $1000$ instâncias cada, isto é, os dados referentes a cada uma das fotografias processadas. Por outro lado, o número de colunas é definido pela quantidade de atributos presentes em cada _dataset_.

## Geração de base de dados tratadas

Dando continuidade ao _pipeline_ de desenvolvimento, com o _dataset_ de $1764$ atributos, foram realizados procedimentos de redução no número de atributos. Esta redução de dimensionalidade se deu para reduzir o efeito da "maldição da dimensionalidade"[^3], onde o aumento exponencial causado pela adição de atributos, também chamados de dimensões, complica o procedimento de ajuste do estimador e/ou não agrega informações relevantes. Desta forma, aplicamos duas técnicas para redução: uma de seleção de atributos através de análise de correlação; e outra de extração de atributos através do _Principal Analysis Component_ (PCA).

No primeiro método de redução de dimensionalidade, a seleção de atributos é feita através de uma medida de similaridade e/ou correlação para detectar atributos redundantes chamada de correlação de Pearson. Nesse caso, a medida indica o grau de associação linear entre duas variáveis quantitativas. Este índice apresenta valores entre $-1$ e $1$, onde $-1$ significa correlação negativa perfeita, ou seja, são inversamente proporcionais; $1$ corresponde a uma correlação positiva perfeita entre variáveis; e $0$ mostra que as duas variáveis não dependem linearmente uma da outra.

Como produto dessa aplicação obtivemos um número reduzido de atributos, mais precisamente $540$. Em suma, a correlação de Pearson nos permitiu selecionar apenas os atributos que apresentaram correlação menor do que $0.75$ entre eles.

Já a segunda técnica utilizada para redução de dimensionalidade foi o PCA. Esta forma de extração de características é um método multivariado muito difundido para redução de dados. O PCA é capaz de sintetizar as informações em um novo conjunto de atributos menor do que o original, porém sem grandes perdas de informações. Costumeiramente, as variáveis numa base de dados são correlacionadas e possuem redundância. De forma simplificada a avaliação de redundância entre dimensões é feita pela análise da matriz de covariância dos dados, elencando de forma decrescente os chamados componentes principais. 

A partir da aplicação do PCA foi gerado uma base com $459$ atributos, onde $98\%$ da informação presente no conjunto inicial foi preservada.

Ao final, contamos com $4$ _datasets_, sendo eles: os dois originais com $1764$ atributos e $900$ atributos, obtidos com o HOG; e outros dois resultantes da aplicação da Correlação de Pearson e do PCA na base de dados com $1764$ atributos. Com isso, concluímos a parte de pré-processamento do _pipeline_. Até aqui foram realizadas tarefas de limpeza, transformação e redução dos dados de modo que eles estejam aptos para a próxima etapa que envolve a utilização dos modelos de aprendizado de máquina.

A implementação realizada para a construção dos _datasets_ pode ser encontrado neste repositório[^4].

## Treinamento

Foram testados 4 algoritmos de classificação, a saber:
- Vizinhos mais próximos (KNN)
- Árvore de decisão (DT)
- Naive bayes gaussiano (GNB)
- Rede neural (MLP)

Além disso, testaram-se diferentes técnicas de comitês de classificação:
- Bagging
- AdaBoost
- Random forest
- Stacking

### Técnicas de amostragem
Uma etapa de grande importância durante a execução do trabalho foi a parte de seleção de amostras para treinamento e testes dos modelos avaliados. Essa fase nos permitiu avaliar se os algoritmos selecionados tinham uma boa capacidade de generalização diante dos dados aos quais foram submetidos. Para isso aplicamos dois métodos de validação cruzada onde os dados são divididos em subconjuntos mutuamente exclusivos chamados de treinamento e teste.

O primeiro método utilizado foi o chamado _holdout_. Como dito anteriormente, ele consiste na divisão dos dados em dois grupos mutuamente exclusivos, onde uma parte serve para o treinamento e outra para o teste. As proporções utilizadas nessa etapa foram $70/30$, $80/20$ e $90/10$; ou seja, dividimos, respectivamente, os dados de todos os _datasets_ proporcionalmente em $70\%$ para treinamento e $30\%$ teste, $80\%$ para treinamento e $20\%$ teste e por fim $90\%$ para treinamento e $10\%$ teste.

A segunda técnica de validação que aplicamos foi o $k$-_fold_. 
Esse método consiste em dividir os dados em $k$ partes iguais, em seguida ajusta-se o modelo utilizando $k-1$ partes, e o restante fica destinada à validação. 
No nosso projeto esse processo foi repetido $10$ vezes, ou seja, $k = 10$. Desta forma, selecionamos $10$ amostras, que por sua vez foram divididas em $10$ partes, utilizando $9$ partes para o treinamento e $1$ para o teste. 
Por fim, os resultados foram combinados obtendo a média dos erros obtidos.

Assim, uma amostra é gerada pelos valores de acurácia de uma dada configuração de um classificador, levando em conta $4$ bases com dimensões distintas ($1764$,$900$,$540$,$459$), nas quais, em cada uma delas, é realizada as 4 técnicas de amostragem. Dessa forma, uma única amostra é formada por $16$ valores de acurácia, resultantes da combinação das diferentes bases e técnicas de amostragem.

As implementações dos códigos para a realização dos procedimentos deste trabalho se encontram neste repositório[^5].

## Validação e análise de desempenho

Nesta seção realiza-se uma análise de desempenho dos estimadores treinados. Para isso, foi realizada a análise de alguns parâmetros dos estimadores estudados, para identificar o ajuste de cada algoritmo. 

Portanto, a análise de uma classificador é realizada em dois níveis. Primeiramente, é feita uma anáĺise gráfica das médias obtidas em cada amostra com a variação dos parâmetros. Posteriormente, é realizado testes estatísticos para decidir sobre a melhor configuração daquele classificador.

### Testes estatísticos

Para definir a melhor configuração de um algoritmo pode-se comparar os valores médio de acurácia de amostras geradas por diferentes parâmetros de um mesmo algoritmo, ou ainda pode-se comparar as acurácias médias obtidas por diferentes algoritmos. Entretanto, o que garante que as melhorias de acurácia promovidas por uma determinada configuração são, de fato, relevantes? Além disso, ainda que um determinado algoritmo tenha uma melhor acurácia média do que outro, o que garante que essa melhoria é de fato significativa a ponto de se optar por um em detrimento de outro? Para tomada de decisões nesse sentido, é importante a realização dos testes estatísticos. Vale salientar que foi adotado neste trabalho um nível de significância de $5\%$.

Nesse sentido, dois tipos de testes foram realizados neste trabalho. O primeiro é o teste estatístico de Friedman, que analisa um conjunto de pelo menos três amostras dependentes, partindo da hipótese nula de que o conjunto de amostras testado provêm de uma mesma população. Assim, para um determinado conjunto de amostras, caso o resultado do $p$-valor seja menor do que o nível de significância previamente definido, então a hipótese nula pode ser descartada e é aceita a hipótese alternativa que declara que o conjunto de amostras provêm de populações diferentes. 

Caso a hipótese alternativa seja aceita no teste de Friedman, é necessário realizar ainda um teste estatístico par-a-par entre as amostras do conjunto. Isso é feito através do teste de Nemenyi, que também assume a hipótese nula de que dado par de amostras analisadas provêm da mesma população. Através desse teste obtém-se um $p$-valor para cada par de amostras e, a partir dele, aceita-se ou ou não a hipótese nula, considerando o nível de significância adotado.

### Parâmetros testados

#### KNN

Em nosso trabalho variamos o número de vizinhos entre 1 e 5 com o intuito de averiguar qual número ideal para a solução do problema.

#### DT

Em nossos testes avaliamos o algoritmo de árvores entre 3 e 7 camadas de profundidade para observar qual dessas melhor se adequa ao problema.

#### GNB

Neste algoritmo foi adotados apenas os parâmetros _default_.

#### MLP

Os parâmetros que buscou-se avaliar no trabalho são: função de ativação, número de neurônios na camada escondida, número máximo de iterações e a taxa de aprendizado inicial.

#### Bagging

Foram testados bagging com 10 e 20 estimadores base do tipo: Vizinhos mais próximos, Árvore de decisão, Naive Bayes e Rede Neural. Assim, gerando 8 combinações de configurações possíveis. Os resultados destas combinações podem ser encontrados na próxima seção deste texto.

#### Boosting

Foram testados boosting com 10 e 20 estimadores base do tipo: Árvore de decisão e Naive Bayes. Assim, gerando 4 combinações de configurações possíveis. Os resultados destas combinações podem ser encontrados na próxima seção deste texto.

#### Random forest

Foram testados Random Forests com 10 e 100 estimadores base e dois critérios de seleção de atributos: gini e entropy. Assim, gerando 4 combinações de configurações possíveis. Os resultados destas combinações podem ser encontrados na próxima seção deste texto.

#### Stacking

Foram estudados três configurações distintas para o stacking com $3$, $10$ e $20$ estimadores. Os três estimadores da primeiras configuração eram KNN, GNB e MLP com parâmetros _default_.



A segunda configuração contava com $5$ estimadores base do tipo MLP cujas configurações usadas são exibidas na tabela abaixo e $5$ KNN com números de vizinhos mais próximos variando de $1$ a $5$, totalizando $10$ estimadores base. 

<div style="margin-left: auto;
            margin-right: auto;
            width: 30%">

| Nome | Função de ativação | Número de neurônios |
|------|--------------------|---------------------|
| MLP1 | `logistica'        | 100                 |
| MLP2 | `tanh'             | 100                 |
| MLP3 | `relu'             | 100                 |
| MLP4 | `relu'             | 200                 |
| MLP5 | `relu'             | 300                 |

</div>

A teceira configuração do stacking contava com 9 estimadores base do tipo MLP cujas configurações usadas são exibidas na tabela abaixo, $10$ KNN com números de vizinhos mais próximos variando de $1$ a $10$ e $1$ Naive Bayes com parâmetros padrões, totalizando $20$ estimadores base. 


<div style="margin-left: auto;
            margin-right: auto;
            width: 30%">

| Nome | Função de ativação | Número de neurônios |
|------|--------------------|---------------------|
| MLP1 | `logistica'        | 100                 |
| MLP2 | `tanh'             | 100                 |
| MLP3 | `relu'             | 100                 |
| MLP4 | `logistica'        | 200                 |
| MLP5 | `tanh'             | 200                 |
| MLP6 | `relu'             | 200                 |
| MLP7 | `logistica'        | 300                 |
| MLP8 | `tanh'             | 300                 |
| MLP9 | `relu'             | 300                 |

</div>

Os resultados das análises de desempenho (gráficos de acurácia e testes estatísticos) podem ser encontrados neste repositório[^6].

## Conclusões

Este trabalho propôs-se a resolver um problema de classificação, em que o objetivo era prever cinco classes distintas, em que três delas eram raças de cachorro e duas delas raças de gato. Um total de oito algoritmos distintos de aprendizado de máquina foram testados. Quatros deles foram algoritmos simples, isto é, sem uso de comitês, a saber: vizinhos mais próximos (KNN), árvore de decisão (DT), Naive Bayes (NB) e rede neural (MLP). Além disso, foram considerados quatro comitês de classificação: bagging, boosting, random forest e stacking. Para avaliar e comparar os estimadores citados foi usada a métrica de acurácia.

Primeiramente, foram avaliados diferentes configurações para cada algoritmo simples. O objetivo era encontrar parâmetros que otimizassem a acurácia dos mesmos. Para o KNN, o melhor número de vizinhos encontrado foi de 3. Para o DT, a máxima profundidade igual a 4 foi aquela que resultou em melhor desempenho. O Naive Bayes foi mantido com seus valores padrões. Por último, a configuração ótima para o MLP foi aquela que usou a função de ativação `relu', o número de neurônios do tipo "OA", o número máximo de iterações igual a 1000 e a taxa de aprendizagem inicial de 0.001. 

Posteriormente, foi avaliado os melhores algoritmos entre as configurações ótimas de cada um deles, citadas anteriormente. Dessa análise, concluiu-se através de testes estatísticos que tanto o NB quanto o MLP tiveram desempenhos semelhantes, embora este último tenha apresentado valor médio de acurácia um pouco maior que o primeiro, ainda que não estatisticamente diferente.

Com relação aos comitês foram avaliados três algoritmos distintos: bagging, boosting, random forest e stacking. O melhor desempenho observado entre os comitês foi para o stacking com $3$ estimadores.

Finalmente, realizou-se uma análise comparativa entre o melhor comitê e o melhor classificador simples, onde notou-se que, de fato, a utilização dos comitês resultou em melhoria na acurácia da predição. Em contrapartida, destaca-se o uso de comitês aumenta a complexidade do problema e também o custo computacional envolvido no treinamento do algoritmo. 

Com relação a análise comparativa entre as acurácia relativas a cada base de dados, percebeu-se que, de forma geral, a base de dados com $900$ atributos teve melhor desempenho em relação as demais.

# Referências
[^1]: Link para o repositório: [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/).
[^2]: Dalal, N. and Triggs, B. (2005). Histograms of oriented gradients for human detection. IEEE.
[^3]: Facelli, K., Lorena, A. C., Gama, J., and Carvalho, A. C. P. L. F. (2011). Inteligência artificial: uma abordagem de aprendizado de máquina. LTC.
[^4]: Notebooks com implementação para a construção dos _datasets_: [https://github.com/BrunoRammon/gato-cachorro_classificacao_de_racas/blob/main/building_processed_datasets.ipynb](https://github.com/BrunoRammon/gato-cachorro_classificacao_de_racas/blob/main/building_processed_datasets.ipynb)
[^5]: Notebook com implementações realizadas: [https://github.com/BrunoRammon/gato-cachorro_classificacao_de_racas/blob/main/train_validate_models.ipynb](https://github.com/BrunoRammon/gato-cachorro_classificacao_de_racas/blob/main/train_validate_models.ipynb)
[^6]: Notebook com análises de desempenho: [https://github.com/BrunoRammon/gato-cachorro_classificacao_de_racas/blob/main/models_performance_analysis.ipynb](https://github.com/BrunoRammon/gato-cachorro_classificacao_de_racas/blob/main/models_performance_analysis.ipynb)
