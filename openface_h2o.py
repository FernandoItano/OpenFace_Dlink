import h2o

print("Conectando no cluster H2O")
h2o.init()

print("Importando dados...")
dados = h2o.import_file(path="/home/fernando/PycharmProjects/OpenFace_Dlink/faces2.csv", header=1)
print(dados)

print("Importando Modelo Treinado")
openface_model = h2o.load_model("/home/fernando/PycharmProjects/OpenFace_Dlink/DeepLearning_model_R_1503776146630_952")

print("Estimando Face")
predicao = openface_model.predict(test_data=dados)
print(predicao)