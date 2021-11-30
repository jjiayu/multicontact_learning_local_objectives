import sys

#Collect External Parameters
#   Get Useful External Parameter List
externalParasList = sys.argv[1:]
#   Make the default dict
ExternalParameters = {"WorkingDirectory": None,
                      "Exp_Prefix": "GroupX",
                      "EnvModelPath": None}
print(ExternalParameters)

#   Update External Parameters
for i in range(len(externalParasList)//2):
    #First check if it is the key of the default the dict
    if externalParasList[2*i][1:] in ExternalParameters: #remove the "-" from the parameter list
        ExternalParameters[externalParasList[2*i][1:]] = externalParasList[2*i + 1] #remove the "-" from the parameter list
print(ExternalParameters)