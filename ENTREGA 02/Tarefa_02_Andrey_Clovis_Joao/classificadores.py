import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
 
##########################################DECISION TREE TREINAMENTO##################################################
def train_data_cart():
  from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
  
  # Load the train dataset
  train_data = pd.read_csv("datasets\data_4000v\env_vital_signals.txt", header=None)
  train_data.columns = ['ID', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav', 'classe']

  # Load the test dataset
  validation_data = pd.read_csv("datasets\data_800v\env_vital_signals.txt",  header=None)
  validation_data.columns = ['ID', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav', 'classe']

  features = ['pSist', 'pDiast', 'qPA', 'pulso', 'resp']
  X_train=train_data[features]

  Y_train=train_data['classe']
  X_validation = validation_data[features]
  Y_validation = validation_data['classe']


  tree_classifier = DecisionTreeClassifier(random_state=42,criterion='entropy')
  tree_classifier.fit(X_train,Y_train)

  y_pred = tree_classifier.predict(X_validation)
  accuracy_score = accuracy_score(Y_validation, y_pred)
  print("Accuracy:", accuracy_score)
  print(classification_report(Y_validation, y_pred))
  # cm = confusion_matrix(Y_validation, y_pred, labels=tree_classifier.classes_)
  # disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=tree_classifier.classes_)
  # disp.plot()
  # plt.show()

  # fig = plt.figure(figsize=(8, 6))
  # tree.plot_tree(tree_classifier)
  # plt.show()
  
  #SAVE MODEL
  
  with open('model.pkl','wb') as f:
    pickle.dump(tree_classifier,f)

######################################REALIZA  A CLASSIFICAÇÃO COM BASE NAS VITIMAS RECEBIDIAS##############################
def classification_cart():
  from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
  
  with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

  # Load the test dataset
  validation_data = pd.read_csv("datasets\data_800v\env_vital_signals.txt",  header=None)  #Substituir pela lista de vitimas
  validation_data.columns = ['ID', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav', 'classe']

  features = ['pSist', 'pDiast', 'qPA', 'pulso', 'resp']
  X_validation = validation_data[features]
  Y_validation = validation_data['classe']
  y_pred = clf.predict(X_validation)
  accuracy_score = accuracy_score(Y_validation, y_pred)
  print("Accuracy:", accuracy_score)
  print(classification_report(Y_validation, y_pred))
  
  
##########################################FUZY##################################################
def fuzzy():
  # variáveis linguisticas de entrada e saída
  pSist_range = np.arange(5,22.0,0.1) #pressão sistólica [5,22]
  pDiast_range = np.arange(0,15.0,0.1) #pressão diastólica [5,22]
  qPA_range = np.arange(-10,10.0,0.1) #qualidade de pressão [5,22]
  pulso_range = np.arange(0,200.0,0.1) #pulsação [5,22]
  resp_range = np.arange(0,22.0,0.1) #freq. respiratória [5,22]
  pSist = ctrl.Antecedent(pSist_range, 'pSist')
  pDiast = ctrl.Antecedent(pDiast_range, 'pDiast')
  qPA = ctrl.Antecedent(qPA_range, 'qPA')
  pulso = ctrl.Antecedent(pulso_range, 'pulso')
  resp = ctrl.Antecedent(resp_range, 'resp')

  # variável linguística de saida
  # Metodo de desfuzificação
  # 'centroid', 'mom' (mean of maximum), 'bisector'(2 areas iguais a partir do ponto de max), 'som'(smallest of maximum),
  classe_range = np.arange(0, 5, .1)
  classe = ctrl.Consequent(classe_range, 'classe', defuzzify_method='mom')


  # termos linguísticos para a qualidade de pressão
  qPA['RUIM'] = fuzz.trapmf(qPA.universe, [-10, -10, -5, -2]) + fuzz.trapmf(qPA.universe, [2, 5, 10, 10])
  qPA['BOM'] = fuzz.trimf(qPA.universe, [-3, 0, 3])
  # qPA.view()

  # termos linguísticos para a pulsação
  pulso['BAI'] = fuzz.trapmf(pulso.universe, [0, 0, 50, 70])
  pulso['MED'] = fuzz.trimf(pulso.universe, [60, 100, 120])
  pulso['ALT'] = fuzz.trapmf(pulso.universe, [100, 150, 200, 200])
  # pulso.view()

  # termos linguísticos para a respiração
  resp['BAI'] = fuzz.trapmf(resp.universe, [0, 0, 5, 8])
  resp['MED'] = fuzz.trimf(resp.universe, [6, 9, 14])
  resp['ALT'] = fuzz.trapmf(resp.universe, [12, 15, 22, 22])
  # resp.view()

  # termos linguísticos para a classe gravidade
  classe['CRIT'] = fuzz.trapmf(classe.universe, [0, 0, 1, 1.5])
  classe['INST'] = fuzz.trimf(classe.universe, [1, 2, 2.5])
  classe['P_EST'] = fuzz.trimf(classe.universe, [2, 2.5, 3])
  classe['EST'] = fuzz.trapmf(classe.universe, [2.5, 3, 4, 4])
  # classe.view()
  print(f"{classe.terms.keys()} tam = {len(classe.terms.keys())}")


  # regras
  regras = [ ]
  regras.append(ctrl.Rule(qPA['RUIM'] & pulso['BAI'] & resp['BAI'], classe['CRIT']))
  regras.append(ctrl.Rule(qPA['RUIM'] & pulso['BAI'] & resp['MED'], classe['INST']))
  regras.append(ctrl.Rule(qPA['RUIM'] & pulso['BAI'] & resp['ALT'], classe['CRIT']))

  regras.append(ctrl.Rule(qPA['RUIM'] & pulso['MED'] & resp['BAI'], classe['CRIT']))
  regras.append(ctrl.Rule(qPA['RUIM'] & pulso['MED'] & resp['MED'], classe['P_EST']))
  regras.append(ctrl.Rule(qPA['RUIM'] & pulso['MED'] & resp['ALT'], classe['CRIT']))

  regras.append(ctrl.Rule(qPA['RUIM'] & pulso['ALT'] & resp['BAI'], classe['INST']))
  regras.append(ctrl.Rule(qPA['RUIM'] & pulso['ALT'] & resp['MED'], classe['INST']))
  regras.append(ctrl.Rule(qPA['RUIM'] & pulso['ALT'] & resp['ALT'], classe['CRIT']))

  regras.append(ctrl.Rule(qPA['BOM'] & pulso['BAI'] & resp['BAI'], classe['INST']))
  regras.append(ctrl.Rule(qPA['BOM'] & pulso['BAI'] & resp['MED'], classe['P_EST']))
  regras.append(ctrl.Rule(qPA['BOM'] & pulso['BAI'] & resp['ALT'], classe['INST']))

  regras.append(ctrl.Rule(qPA['BOM'] & pulso['MED'] & resp['BAI'], classe['P_EST']))
  regras.append(ctrl.Rule(qPA['BOM'] & pulso['MED'] & resp['MED'], classe['EST']))
  regras.append(ctrl.Rule(qPA['BOM'] & pulso['MED'] & resp['ALT'], classe['P_EST']))

  regras.append(ctrl.Rule(qPA['BOM'] & pulso['ALT'] & resp['BAI'], classe['INST']))
  regras.append(ctrl.Rule(qPA['BOM'] & pulso['ALT'] & resp['MED'], classe['P_EST']))
  regras.append(ctrl.Rule(qPA['BOM'] & pulso['ALT'] & resp['ALT'], classe['INST']))
  """
  Vamos criar uma representação alternativa das regras para calcularmos os valores de disparo de cada regra, já que o sckfuzzy não deixa acessar diretamente.
  Cada linha representa uma regra. Cada regra faz referência a um termo linguístico. A última coluna diz se é um AND (1) ou um OR (2). 
  """
  regras_alt = [[0,0,0,0,1],
                [0,0,1,1,1],
                [0,0,2,0,1],
                [0,1,0,0,1],
                [0,1,1,2,1],
                [0,1,2,0,1],
                [0,2,0,1,1],
                [0,2,1,1,1],
                [0,2,2,0,1],
                [1,0,0,1,1],
                [1,0,1,2,1],
                [1,0,2,1,1],
                [1,1,0,2,1],
                [1,1,1,3,1],
                [1,1,2,2,1],
                [1,2,0,1,1],
                [1,2,1,2,1],
                [1,2,2,1,1]]

  # # Sistema de fuzzy
  sif_ctrl = ctrl.ControlSystem(regras)
  sif = ctrl.ControlSystemSimulation(sif_ctrl)

  # # Entrada de exemplo 8.733333,134.454047,17.972046
  qPA_input = 8.733333
  pulso_input = 134.454047
  resp_input = 17.972046
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
  print(f"Termo de saída: {termo_saida}\n")

  # plota os termos linguisticos (conjuntos fuzzy) de saída
  # classe.view(sim=sif)



train_data_cart()
classification_cart()