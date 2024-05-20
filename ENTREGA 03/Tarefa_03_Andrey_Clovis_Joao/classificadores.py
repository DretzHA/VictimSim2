import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, mean_squared_error
import csv
import warnings
pd.set_option('future.no_silent_downcasting', True)

warnings.simplefilter(action='ignore', category=UserWarning)

##########################################DECISION TREE TREINAMENTO##################################################
def train_data_cart():
  from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

  # Load the train dataset
  train_data = pd.read_csv("datasets\\data_4000v\\env_vital_signals.txt", header=None)
  train_data.columns = ['ID', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav', 'classe']
  features = ['qPA', 'pulso', 'resp'] #features do treinamento

  X = train_data[features]
  Y = train_data['classe']
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
  # Parameters' definition
  parameters = {
      'criterion': ['entropy'],
      'max_depth': [10, 20, 100],
      'min_samples_leaf': [1, 4, 8]
  }

  tree_classifier = DecisionTreeClassifier(random_state=42)
  # grid search using cross-validation
  # cv = 3 is the number of folds
  # scoring = 'f' the metric for chosing the best model
  clf = GridSearchCV(tree_classifier, parameters, cv=5, verbose=4, return_train_score=True)
  clf.fit(X_train, y_train)

  results_clf = clf.cv_results_
  print(results_clf)
  with open('resultados_classificador.csv','w') as f:
    w = csv.writer(f)
    w.writerows(results_clf.items())
  # the best tree according to the f1 score
  best = clf.best_estimator_
  print("\n* Melhor classificador *")
  print(clf.best_estimator_)
# Predicoes
# com dados do treinamento
  y_pred_train = best.predict(X_train)
  acc_train = accuracy_score(y_train, y_pred_train) * 100
  print(f"Acuracia com dados de treino: {acc_train:.2f}%")

  # com dados de teste (nao utilizados no treinamento/validacao)
  y_pred_test = best.predict(X_test)
  acc_test = accuracy_score(y_test, y_pred_test) * 100
  print(f"Acuracia com dados de teste: {acc_test:.2f}%")


  # cm = confusion_matrix(Y_validation, y_pred, labels=tree_classifier.classes_)
  # disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=tree_classifier.classes_)
  # disp.plot()
  # plt.show()

  # fig = plt.figure(figsize=(8, 6))
  # tree.plot_tree(tree_classifier)
  # plt.show()



  #SAVE MODEL

  with open('model.pkl','wb') as f:
    pickle.dump(best,f)

######################################REALIZA  A CLASSIFICAÇÃO COM BASE NAS VITIMAS RECEBIDIAS##############################
def classification_cart(validation_data):
  from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

  with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

  # Load the test dataset
  #validation_data = pd.read_csv("datasets\data_800v\env_vital_signals.txt",  header=None)  #Substituir pela lista de vitimas
  #validation_data.columns = ['ID', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav', 'classe']
  features = ['qPA', 'pulso', 'resp']
  X_validation = validation_data[features]
  #Y_validation = validation_data['classe']
  y_pred = clf.predict(X_validation)
  validation_data['classe'] = y_pred

  return validation_data

  # accuracy_score = accuracy_score(Y_validation, y_pred)
  # print("Accuracy:", accuracy_score)
  # print(classification_report(Y_validation, y_pred))


##########################################FUZY##################################################
def fuzzy(validation_data):
  # variáveis linguisticas de entrada e saída
  pSist_range = np.arange(5,22.0,0.1) #pressão sistólica [5,22]
  pDiast_range = np.arange(0,15.0,0.1) #pressão diastólica [5,22]
  qPA_range = np.arange(-10,10.0,1) #qualidade de pressão [5,22]
  pulso_range = np.arange(0,200.0,1) #pulsação [5,22]
  resp_range = np.arange(0,22.0,1) #freq. respiratória [5,22]
  #pSist = ctrl.Antecedent(pSist_range, 'pSist')
  #pDiast = ctrl.Antecedent(pDiast_range, 'pDiast')
  qPA = ctrl.Antecedent(qPA_range, 'qPA')
  pulso = ctrl.Antecedent(pulso_range, 'pulso')
  resp = ctrl.Antecedent(resp_range, 'resp')

  # variável linguística de saida
  # Metodo de desfuzificação
  # 'centroid', 'mom' (mean of maximum), 'bisector'(2 areas iguais a partir do ponto de max), 'som'(smallest of maximum),
  classe_range = np.arange(0, 100, 1)
  classe = ctrl.Consequent(classe_range, 'classe', defuzzify_method='centroid')

  # termos linguísticos para a qualidade de pressão
  qPA['BAI'] = fuzz.trapmf(qPA.universe, [-10, -10, -5, -2])
  qPA['MED'] = fuzz.trimf(qPA.universe, [-4, 0, 4])
  qPA['ALT'] = fuzz.trapmf(qPA.universe, [2, 5, 10, 10])
  #qPA.view()

  # termos linguísticos para a pulsação
  pulso['BAI'] = fuzz.trapmf(pulso.universe, [0, 0, 50, 70])
  pulso['MED'] = fuzz.trimf(pulso.universe, [60, 85, 110])
  pulso['ALT'] = fuzz.trapmf(pulso.universe, [100, 150, 200, 200])
  #pulso.view()

  # termos linguísticos para a respiração
  resp['BAI'] = fuzz.trimf(resp.universe, [0, 5.5, 13])
  resp['MED'] = fuzz.trimf(resp.universe, [10, 12, 16.5])
  resp['ALT'] = fuzz.trimf(resp.universe, [14, 18, 22])
  # resp.view()

  # termos linguísticos para a classe gravidade
  classe['CRIT'] = fuzz.trapmf(classe.universe, [0, 0, 20, 25])
  classe['INST'] = fuzz.trimf(classe.universe, [20, 40, 55])
  classe['P_EST'] = fuzz.trimf(classe.universe, [45, 60, 80])
  classe['EST'] = fuzz.trapmf(classe.universe, [75, 90, 100, 100])
  #classe.view()
  #print(f"{classe.terms.keys()} tam = {len(classe.terms.keys())}")


  # regras
  regras = [ ]
  regras.append(ctrl.Rule(qPA['BAI'] & pulso['BAI'] & resp['BAI'], classe['CRIT']))  #1
  regras.append(ctrl.Rule(qPA['MED'] & pulso['BAI'] & resp['BAI'], classe['INST']))  #2
  regras.append(ctrl.Rule(qPA['ALT'] & pulso['BAI'] & resp['BAI'], classe['CRIT']))  #3

  regras.append(ctrl.Rule(qPA['BAI'] & pulso['MED'] & resp['BAI'], classe['INST'])) #4
  regras.append(ctrl.Rule(qPA['MED'] & pulso['MED'] & resp['BAI'], classe['P_EST']))  #5
  regras.append(ctrl.Rule(qPA['ALT'] & pulso['MED'] & resp['BAI'], classe['INST'])) #6

  regras.append(ctrl.Rule(qPA['BAI'] & pulso['ALT'] & resp['BAI'], classe['CRIT']))#7
  regras.append(ctrl.Rule(qPA['MED'] & pulso['ALT'] & resp['BAI'], classe['INST'])) #8
  regras.append(ctrl.Rule(qPA['ALT'] & pulso['ALT'] & resp['BAI'], classe['CRIT'])) #9

  regras.append(ctrl.Rule(qPA['BAI'] & pulso['BAI'] & resp['MED'], classe['INST'])) #10
  regras.append(ctrl.Rule(qPA['MED'] & pulso['BAI'] & resp['MED'], classe['INST'])) #11
  regras.append(ctrl.Rule(qPA['ALT'] & pulso['BAI'] & resp['MED'], classe['INST'])) #12

  regras.append(ctrl.Rule(qPA['BAI'] & pulso['MED'] & resp['MED'], classe['P_EST'])) #13
  regras.append(ctrl.Rule(qPA['MED'] & pulso['MED'] & resp['MED'], classe['EST'])) #14
  regras.append(ctrl.Rule(qPA['ALT'] & pulso['MED'] & resp['MED'], classe['P_EST'])) #15

  regras.append(ctrl.Rule(qPA['BAI'] & pulso['ALT'] & resp['MED'], classe['P_EST'])) #16
  regras.append(ctrl.Rule(qPA['MED'] & pulso['ALT'] & resp['MED'], classe['INST']))#17
  regras.append(ctrl.Rule(qPA['ALT'] & pulso['ALT'] & resp['MED'], classe['INST'])) #18

  regras.append(ctrl.Rule(qPA['BAI'] & pulso['BAI'] & resp['ALT'], classe['INST'])) #19
  regras.append(ctrl.Rule(qPA['MED'] & pulso['BAI'] & resp['ALT'], classe['P_EST']))  #20
  regras.append(ctrl.Rule(qPA['ALT'] & pulso['BAI'] & resp['ALT'], classe['INST']))  #21

  regras.append(ctrl.Rule(qPA['BAI'] & pulso['MED'] & resp['ALT'], classe['P_EST'])) #22
  regras.append(ctrl.Rule(qPA['MED'] & pulso['MED'] & resp['ALT'], classe['EST']))  #23
  regras.append(ctrl.Rule(qPA['ALT'] & pulso['MED'] & resp['ALT'], classe['P_EST']))  #24

  regras.append(ctrl.Rule(qPA['BAI'] & pulso['ALT'] & resp['ALT'], classe['INST'])) #25
  regras.append(ctrl.Rule(qPA['MED'] & pulso['ALT'] & resp['ALT'], classe['P_EST'])) #26
  regras.append(ctrl.Rule(qPA['ALT'] & pulso['ALT'] & resp['ALT'], classe['INST'])) #27
  """
  Vamos criar uma representação alternativa das regras para calcularmos os valores de disparo de cada regra, já que o sckfuzzy não deixa acessar diretamente.
  Cada linha representa uma regra. Cada regra faz referência a um termo linguístico. A última coluna diz se é um AND (1) ou um OR (2). 
  """
  regras_alt = [[0,0,0,0,1], #1
                [1,0,0,1,1], #2
                [2,0,0,0,1], #3
                [0,1,0,1,1], #4
                [1,1,0,2,1], #5
                [2,1,0,1,1],  #6
                [0,2,0,0,1],#7
                [1,2,0,1,1], #8
                [2,2,0,0,1], #9
                [0,0,1,1,1], #10
                [1,0,1,1,1], #11
                [2,0,1,1,1], #12
                [0,1,1,2,1], #13
                [1,1,1,3,1], #14
                [2,1,1,2,1], #15
                [0,2,1,2,1], #16
                [1,2,1,1,1], #17
                [2,2,2,1,1], #18
                [0,0,2,1,1], #19
                [1,0,2,2,1], #20
                [2,0,2,1,1], #21
                [0,1,2,2,1], #22
                [1,1,2,3,1], #23
                [2,1,2,2,1], #24
                [0,2,2,1,1], #25
                [1,2,2,2,1], #26
                [2,2,2,1,1]] #27

  # # Sistema de fuzzy
  sif_ctrl = ctrl.ControlSystem(regras)
  sif = ctrl.ControlSystemSimulation(sif_ctrl)

  # # Entrada dos dados das vitimas
  res_classe = []
  qPA_df =validation_data['qPA']
  pulso_df =validation_data['pulso']
  resp_df =validation_data['resp']
  for i in range(0,len(validation_data)):
    qPA_input = qPA_df[i]
    pulso_input = pulso_df[i]
    resp_input = resp_df[i]
    sif.input['qPA'] = qPA_input
    sif.input['pulso'] = pulso_input
    sif.input['resp'] = resp_input

    # # Computa a saída
    sif.compute()

    #GRAUS DE PERTINÊNCIA
    # print(f"\n*** ENTRADAS ***")
    # print(f"Graus de pertinência da entrada qualidade de pressão = {qPA_input}")
    mi_qPA =  [0] * len(qPA.terms.keys())
    i = 0
    for termo, fuzzy_set in qPA.terms.items():
        mi_qPA[i] = fuzz.interp_membership(qPA.universe, qPA[termo].mf, qPA_input)
        # print(f"  ao termo {termo} = {mi_qPA[i]:.3f}")
        i = i + 1

    # # Plotar o grau de pertinência
    # qPA.view(sim=sif)

    # print(f"Graus de pertinência da entrada pulso = {pulso_input}")
    mi_pulso =  [0] * len(pulso.terms.keys())
    i = 0
    for termo, fuzzy_set in pulso.terms.items():
        mi_pulso[i] = fuzz.interp_membership(pulso.universe, pulso[termo].mf, pulso_input)
        # print(f"  ao termo {termo} = {mi_pulso[i]:.3f}")
        i = i + 1

    # # Plotar o grau de pertinência
    # pulso.view(sim=sif)

    # print(f"Graus de pertinência da entrada respiração = {resp_input}")
    mi_resp =  [0] * len(resp.terms.keys())
    i = 0
    for termo, fuzzy_set in resp.terms.items():
        mi_resp[i] = fuzz.interp_membership(resp.universe, resp[termo].mf, resp_input)
        # print(f"  ao termo {termo} = {mi_resp[i]:.3f}")
        i = i + 1


    nivel_disparo = [0] * len(regras_alt)          # nivel de disparo por regra
    nivel_agregado = [0] * len(classe.terms.keys()) # maior nivel de disparo por termo de saída
    for i, r_alt in enumerate(regras_alt):
      if r_alt[4] == 1:    # AND
        nivel_disparo[i] = min(mi_qPA[r_alt[0]], mi_pulso[r_alt[1]], mi_resp[r_alt[2]])
      else: # OR
        nivel_disparo[i] = max(mi_qPA[r_alt[0]], mi_pulso[r_alt[1]], mi_resp[r_alt[2]])

      if nivel_disparo[i] > nivel_agregado[r_alt[3]]:
        nivel_agregado[r_alt[3]] = nivel_disparo[i]

    # print(f"\n*** Nível de disparo por regra ***")
    # for i, r in enumerate(sif_ctrl.rules):
    #   print(f"Regra {i}: {r.antecedent} ==> {r.consequent} {nivel_disparo[i]:.3f}")

    # print(f"\n*** Agregação dos consequentes das regras ***")
    # for i, termo in enumerate(classe.terms):
    #   print(f"{termo} = {nivel_agregado[i]:.3f}")


    # print(f"\nValor de qPA = {qPA_input}, pulso = {pulso_input} e respiração = {resp_input}")

    # Obter o valor desfuzzificado
    output = sif.output['classe']
    # print(f"Valor da saida do SIF para a classe de gravidade = {output:.3f} desfuzzificada por {classe.defuzzify_method}")

    # Obter o termo de saida mais relevante a partir do valor desfuzzificado
    termo_saida = max(classe.terms.keys(), key=lambda term: fuzz.interp_membership(classe.universe, classe[term].mf, output))
    #print(f"Termo de saída: {termo_saida}\n")
    if termo_saida =='CRIT':
      res_classe.append(1)
    elif termo_saida =='INST':
      res_classe.append(2)
    elif termo_saida =='P_EST':
      res_classe.append(3)
    else:
      res_classe.append(4)

  validation_data['classe'] = res_classe
  return validation_data

    # plota os termos linguisticos (conjuntos fuzzy) de saída
    # classe.view(sim=sif)


def dict2df(victims_dict):
  i = 0
  for victims_per_agent in victims_dict:
    df_victims = pd.DataFrame(columns=['ID', 'posX', 'posY', 'qPA', 'pulso', 'resp', 'classe', 'grav']) #cria dataframe
    for key, value in victims_per_agent.items():
        '''passa valores do dict das vitimas para o dataframe'''
        id_victim = key
        posx_victim = value[0][0]
        posy_victim = value[0][1]
        qPA_victim = value[1][3]
        pulso_victim = value[1][4]
        resp_victim = value[1][5]
        new_row = [id_victim, posx_victim, posy_victim, qPA_victim, pulso_victim, resp_victim, 0, 0] #new row to append in dataframe
        df_victims.loc[len(df_victims)] = new_row #append row in datafram
        df_victims['ID']=df_victims['ID'].astype(int)
        df_victims['posX']=df_victims['posX'].astype(int)
        df_victims['posY']=df_victims['posY'].astype(int)

    df_victims = classification_cart(df_victims) #manda dataframe para funcao que vai realizar a classificao por arvore de ddecisao
    df_victims = test_neural_regressor_grav(df_victims) #manda dataframe para funcao que vai realizara gravidade por regressao MLP

    resultado_csv = pd.DataFrame(columns=['ID', 'x', 'y', 'grav', 'classe']) #cria dataframe para gravarr resultados
    resultado_csv['ID'] = df_victims['ID']
    resultado_csv['x'] = df_victims['posX']
    resultado_csv['y'] = df_victims['posY']
    resultado_csv['classe'] = df_victims['classe']
    resultado_csv['grav'] = df_victims['grav']
    resultado_csv.fillna(0, inplace=True)
    if i ==0:
      resultado_csv.to_csv("clusters\\cluster1.txt", header=False, index=False) #salva resultados cluster1 (vitimas grupo 1)
    elif i ==1:
      resultado_csv.to_csv("clusters\\cluster2.txt", header=False, index=False) #salva resultados cluster2 (vitimas grupo 2)
    elif i ==2:
      resultado_csv.to_csv("clusters\\cluster3.txt", header=False, index=False) #salva resultados cluster3 (vitimas grupo 3)
    else:
      resultado_csv.to_csv("clusters\\cluster4.txt", header=False, index=False) #salva resultados cluster4 (vitimas grupo 4)
    i+=1

    j=0
    for key, value in victims_per_agent.items():
      '''coloca o valor de gravidade para dentro do dict das vitimas'''
      classe = df_victims.iloc[j,6]
      grav = df_victims.iloc[j,7]
      value[1].append(classe)
      value[1].append(grav)
      j+=1


  return victims_dict


#############################################################################################################################


def train_neural_regressor_grav():
  # Load the train dataset
  train_data = pd.read_csv("datasets\\data_4000v\\env_vital_signals.txt", header=None)
  train_data.columns = ['ID', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav', 'classe']
  features = ['qPA', 'pulso', 'resp']  # features do treinamento

  X = train_data[features]
  Y = train_data['grav']
  # split train and test dataset
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

  parameters = {
    'hidden_layer_sizes': [(200,), (100,), (50,), (64, 32, 16)],
    'activation': ['relu', 'logistic', 'identity']
  }

  # MLP settings
  nn = MLPRegressor(random_state=42, max_iter=1000)
  # grid search using cross-validation
  # cv = 3 is the number of folds
  # scoring = 'f' the metric for chosing the best model
  nn_grav = GridSearchCV(nn, parameters, cv=5, scoring='neg_root_mean_squared_error')
  nn_grav.fit(X_train, y_train)

  results_nn = nn_grav.cv_results_
  print(results_nn)
  with open('resultados_gravidade_gridsearch.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(results_nn.items())
  # the best tree according to the f1 score
  best = nn_grav.best_estimator_
  print('* Melhor classificador *')
  print(nn_grav.best_estimator_)

  best.fit(X_train, y_train)  # train data with best estimator

  # # Make prediction
  pred = best.predict(X_test)
  # # Calculate accuracy and error metrics
  test_set_rmse = np.sqrt(mean_squared_error(y_test, pred))
  # # Print R_squared and RMSE value
  print("Gravidade:")
  print('RMSE: ', test_set_rmse)

  # SAVE MODEL

  with open('model_gravidade.pkl', 'wb') as f:
    pickle.dump(best, f)

  # PLOT LEARNING CURVES
  #best = MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=1000, random_state=42)
  train_errors, test_errors = [], []

  for m in range(1, len(X_train), 20):
    best.fit(X_train[:m], y_train[:m])
    y_train_pred = best.predict(X_train[:m])
    y_test_pred = best.predict(X_test)
    train_errors.append(mean_squared_error(y_train[:m], y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

  plt.plot(np.sqrt(train_errors), 'r', label='train')
  plt.plot(np.sqrt(test_errors), 'b', label='test')
  plt.title('Curvas de aprendizagem da Rede Neural de Gravidade')
  plt.show()


def train_neural_regressor_prior():
  # Load the train dataset
  train_data = pd.read_csv("datasets\\data_300v_90x90\\rescuer_prior.txt", header=None)
  train_data.columns = ['x1', 'x2', 'x3', 'x4', 'p']
  features = ['x1', 'x2', 'x3', 'x4'] #features do treinamento

  X = train_data[features]
  Y = train_data['p']
  #split train and test dataset
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
  #MLP settings
  parameters = {
      'hidden_layer_sizes': [(200,),(100,),(50,)],
      'activation': ['relu', 'logistic', 'identity']
  }

  #MLP settings
  nn = MLPRegressor(random_state=42, max_iter=1000)
  # grid search using cross-validation

  nn_prior = GridSearchCV(nn, parameters, cv=5, scoring='neg_root_mean_squared_error')
  nn_prior.fit(X_train, y_train)

  results_nn = nn_prior.cv_results_
  print(results_nn)
  with open('resultados_prioridade_gridsearch.csv','w') as f:
    w = csv.writer(f)
    w.writerows(results_nn.items())
  # the best tree according to the f1 score
  best = nn_prior.best_estimator_
  print("\n* Melhor classificador *")
  print(nn_prior.best_estimator_)

  best.fit(X_train, y_train) #train data

  # # Make prediction
  pred = best.predict(X_test)
  # # Calculate accuracy and error metrics
  test_set_rmse = np.sqrt(mean_squared_error(y_test, pred))
  # # Print R_squared and RMSE value
  print("Prioridade:")
  #print('R_squared value: ', test_set_rsquared)
  print('RMSE: ', test_set_rmse)

  #SAVE MODEL

  with open('model_prioridade.pkl','wb') as f:
    pickle.dump(best,f)

    # PLOT LEARNING CURVES

  # best = MLPRegressor(activation='logistic', hidden_layer_sizes=(200,), max_iter=1000,
  #                random_state=42)

  # train_errors, test_errors = [], []

  # for m in range(1, len(X_train), 20):
  #   best.fit(X_train[:m], y_train[:m])
  #   y_train_pred = best.predict(X_train[:m])
  #   y_test_pred = best.predict(X_test)
  #   train_errors.append(mean_squared_error(y_train[:m], y_train_pred))
  #   test_errors.append(mean_squared_error(y_test, y_test_pred))

  # plt.plot(np.sqrt(train_errors), 'r', label='train')
  # plt.plot(np.sqrt(test_errors), 'b', label='test')
  # plt.title('Curvas de aprendizagem da Rede Neural de Prioridade')
  # plt.show()


def test_neural_regressor_grav(data):

  #original_data = data.copy(deep=True) #copia original para RMSE
  features = ['qPA', 'pulso', 'resp']
  X_validation = data[features]
  y_pred = model_grav.predict(X_validation)
  data['grav'] = y_pred


   ############PARA VERIFICAR RMSE COM DADO DE TESTE, DESCOMENTAR. PARA RODAR SISTEMA MULTI AGENTE, DEIXAR COMENTADO###########################
  # original_data = pd.read_csv("datasets\\data_800v\\env_vital_signals.txt",  header=None) # ler dados
  # original_data.columns = ['ID', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav', 'classe'] #atribui as colunas ao DF
  # test_set_rmse = np.sqrt(mean_squared_error(original_data['grav'], y_pred))
  # # Print R_squared and RMSE value
  # print("Gravidade:")
  # print('RMSE: ', test_set_rmse)

  return data


def test_neural_regressor_prior(data):
  features = ['x1', 'x2', 'x3', 'x4']
  X_validation = data[features]
  y_pred = model_prior.predict(X_validation)
  data['p'] = y_pred

  ############PARA VERIFICAR RMSE COM DADO DE TESTE, DESCOMENTAR. PARA RODAR SISTEMA MULTI AGENTE, DEIXAR COMENTADO###########################

  # original_data = pd.read_csv("datasets\\data_300v_90x90\\rescuer_prior_preblind_target.txt",  header=None) # dataset com resultados para RMSE
  # original_data.columns = ['x1', 'x2', 'x3', 'x4', 'p'] #atribui as colunas ao DF
  # test_set_rmse = np.sqrt(mean_squared_error(original_data['p'], y_pred))
  # # Print R_squared and RMSE value
  # print("Prioridade:")
  # print('RMSE: ', test_set_rmse)

  return data


def priority_calculus(X):

  y_pred = model_prior.predict(X)
  return np.sqrt(np.sum(np.array(y_pred))/(len(X)*100))


########################################################PARA REALIZAR O TESTE, BASTA COLOCAR O CAMINHO DO ARQUIVO####################
# data_grav = pd.read_csv("datasets\\data_800v\\env_vital_signals.txt",  header=None) # ler dados
# data_grav.columns = ['ID', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav', 'classe'] #atribui as colunas ao DF

# data_prior = pd.read_csv("datasets\\data_300v_90x90\\rescuer_prior_preblind.txt",  header=None) # ler dados prioridades
# data_prior.columns = ['x1', 'x2', 'x3', 'x4'] #atribui as colunas ao DF prioridades

#train_data_cart() #funcao treinamento do modelo de classificação
#train_neural_regressor_grav() #funcao de treinamento regressão MLP
#train_neural_regressor_prior() #funcao de treinamento regressão prioridade


#resultado = fuzzy(data_grav)
# resultado = classification_cart(data_grav) #realiza a classificacao com base nos dados por CART
# resultado_csv = pd.DataFrame(columns=['ID', 'x', 'y', 'grav', 'classe']) #datarame do resultado das classes
# resultado_csv['ID'] = resultado['ID']
# resultado_csv['classe'] = resultado['classe']
# resultado_csv.fillna(0, inplace=True)
#print(resultado_csv)
# resultado_csv.to_csv("pred.txt", header=False, index=False) #salvar arquivo com predição de classes

f = open('model_prioridade.pkl', 'rb')
model_prior = pickle.load(f)

f = open('model_gravidade.pkl', 'rb')
model_grav = pickle.load(f)

f = open('model.pkl', 'rb')
model_cart = pickle.load(f)

# test_neural_regressor_grav(data_grav) #testa regressor de gravidades
# test_neural_regressor_prior(data_prior) #testa regressor das prioridades

