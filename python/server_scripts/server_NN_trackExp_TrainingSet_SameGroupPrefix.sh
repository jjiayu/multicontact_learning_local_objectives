#External Inputs
#$1 --- Working Direcotry
#$2 --- Folder name for Saving the Experiments
#$3 --- Folder name for the one Saves Ground Truth Training Set
#$4 --- Prefix of the Target Experiment File name (That will be loaded for Envrionments)
#$5 --- Prefix of the Exp File name (For saved files)
#$6 --- The path of Machine learning folder (NN_Model_Valid)

#Command Example (On server):
#GroupName=Group20 && longjob -28day -c "nice bash server_NN_trackExp_TrainingSet_SameGroupPrefix.sh Rubbles_DaggerExact NN_TrackTraining_InitSet_Dagger_2Iter CleanTrainingSetRollOuts_InitialSet $GroupName $GroupName /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/Rubbles_DaggerExact/ML_Models/NN_Model_Valid_OriginalForm_Dagger_InitSet_2Iter/"
#GroupName=Group20 && longjob -28day -c "nice bash server_NN_trackExp_TrainingSet_SameGroupPrefix.sh Rubbles_RegretOneStep NN_TrackValidation_Aug_1StepBeforeFail_1Time CleanValidationSetRollOuts $GroupName $GroupName /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/Rubbles_RegretOneStep/ML_Models/NN_Model_Valid_OriginalForm_Aug_1StepBeforeFail_1Time/"

#GroupName=Group20 && longjob -28day -c "nice bash server_NN_trackExp_TrainingSet_SameGroupPrefix.sh Rubbles_Add1Step_1to2StepsbeforeFail_OutlierClip NN_TrackTrainingAll_Add1Step_1to2StepbeforeFail_2Time_RemovebyClip CleanTrainingSetRollOuts_All $GroupName $GroupName /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/Rubbles_Add1Step_1to2StepsbeforeFail_OutlierClip/ML_Models/NN_Model_Aug1Step_1to2StepbeforeFail_2Time_RemovebyClip"
#GroupName=Group13 && longjob -28day -c "nice bash server_NN_trackExp_TrainingSet_SameGroupPrefix.sh Rubbles_Add2Steps_1StepbeforeFail_RemovebyClip NN_TrackTrainingAll_Add2Steps_1StepbeforeFail_1Time_RemovebyClip CleanTrainingSetRollOuts_All $GroupName $GroupName /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/Rubbles_Add2Steps_1StepbeforeFail_RemovebyClip/ML_Models/NN_Model_Aug2Steps_1StepbeforeFail_1Time_RemoveOutlier"

#GroupName=Group20 && longjob -28day -c "nice bash server_NN_trackExp_TrainingSet_SameGroupPrefix.sh Rubbles_Noise_Add2Steps_1StepbeforeFail_RemovebyClip NN_TrackTrainingAll_InitialSet_Noise CleanTrainingSetRollOuts_All $GroupName $GroupName /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/Rubbles_Noise_Add2Steps_1StepbeforeFail_RemovebyClip/ML_Models/NN_Model_InitialSet_Noise"
#GroupName=Group20 && longjob -28day -c "nice bash server_NN_trackExp_TrainingSet_SameGroupPrefix.sh Rubbles_Add1Step_1to2StepsbeforeFail_RemovebyClip NN_TrackTrainingAll_Add1Step_1to2StepbeforeFail_1Time_RemovebyClip CleanTrainingSetRollOuts_All $GroupName $GroupName /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/Rubbles_Add1Step_1to2StepsbeforeFail_RemovebyClip/ML_Models/NN_Model_Aug1Step_1to2StepbeforeFail_1Time_RemovebyClip"

#GroupName=Group20 && longjob -28day -c "nice bash server_NN_trackExp_TrainingSet_SameGroupPrefix.sh Rubbles_AddVarSteps_1StepbeforeFail_RemovebyClip NN_TrackTrainingAll_AddVarSteps_1StepbeforeFail_1Time_RemovebyClip_SmallThre CleanTrainingSetRollOuts_All $GroupName $GroupName /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/Rubbles_AddVarSteps_1StepbeforeFail_RemovebyClip/ML_Models/NN_Model_AugVarStep_1StepbeforeFail_1Time_RemovebyClip_SmallThre"

#GroupName=Group20 && longjob -28day -c "nice bash server_NN_trackExp_TrainingSet_SameGroupPrefix.sh Rubbles_AddVarSteps_1to2StepbeforeFail_RemovebyClip NN_TrackTrainingAll_AddVarSteps_1to2StepbeforeFail_3Time_RemovebyClip_SmallThre CleanTrainingSetRollOuts_All $GroupName $GroupName /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/Rubbles_AddVarSteps_1to2StepbeforeFail_RemovebyClip/ML_Models/NN_Model_AugVarStep_1to2StepbeforeFail_3Time_RemovebyClip_SmallThre"

#GroupName=Group1 && longjob -28day -c "nice bash server_NN_trackExp_TrainingSet_SameGroupPrefix.sh LargeSlope_Angle_23_X_negative NN_TrackTrainingAll_LargeSlope_InitialSet Clean_Training_RollOuts_LargeSlope_20Steps_Start12_Large5_Angle23_X_negative_4Steps $GroupName $GroupName /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/LargeSlope_Angle_23_X_negative/ML_Models/NN_Model_InitialSet"


#---------------------
#Get the path of the Working Directory
cd /afs/inf.ed.ac.uk/group/project/mlp_localobj/
#cd /home/jiayu/Desktop/MLP_DataSet/
#mkdir $1
cd $1
pwd
filedir=$PWD

echo $filedir

compute_rounds=1

#Get the path of Training RollOuts
rootfolder=/afs/inf.ed.ac.uk/group/project/mlp_localobj/ 
#rootfolder=/home/jiayu/Desktop/MLP_DataSet/ 
workingfolder=$1
slash=/
rooloutfolder=$3 #CleanTrainingSetRollOuts/
cd $rootfolder$workingfolder$slash$rooloutfolder$slash

#get number of exp
numExp=$(ls ${4}_*.p | wc -l)

#Run the Optimization
for i in $4_*.p; do #Random loop control, just loop over all the files
    
    echo "Iteration: " $compute_rounds

   #We use randomly sampled files
    #tempfilename=$(ls -1 *.p | shuf -n 1)
    #envfile=$(realpath $tempfilename)
    #echo $envfile

   #Or we use the first few files from the list
    echo "$i"
    envfile=$(realpath $i)

    #python3 /home/jiayu/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \ 
    python3 -W ignore /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \
    -WorkingDirectory $filedir \
    -RollOutFolderName $2 \
    -Exp_Prefix $5 \
    -EnvModelPath $envfile \
    -LocalObjTrackingFlag Yes \
    -LocalObj_from NeuralNetwork \
    -ML_ModelPath $6

    let compute_rounds=compute_rounds+1

   if  [[ $compute_rounds -gt $numExp ]]; then
      break
   fi

done