#External Inputs
#$1 --- Working Direcotry
#$2 --- Folder name for Saving the Experiments
#$3 --- Folder name for the one Saves Ground Truth Training Set
#$4 --- Prefix of the Target Experiment File name (That will be loaded for Envrionments)
#$5 --- Prefix of the Exp File name (For saved files)
#$6 --- Noise Level


#Command Example (On server):

#GroupName=Group1 && bash Desktop_LookUp_TimingObj_trackExp_TrainingSet_SameGroupPrefix.sh LookUp_LargeSlope_Angle_21_26 LookUp_Tracking_NoFixing CleanRollOuts_LargeSlope_20Steps_Start12_Large5_Angle21_26_2Steps $GroupName $GroupName 0.0 

#GroupName=Group1 && bash Desktop_LookUp_TimingObj_trackExp_TrainingSet_SameGroupPrefix.sh LargeSlope_Angle_23_X_negative LookUp_Tracking_2Steps_NoFixing Clean_RollOuts_LargeSlope_Start12_Large5_X_negative_Angle23_2Steps $GroupName $GroupName 0.0 

#GroupName=Group1 && bash Desktop_LookUp_TimingObj_trackExp_TrainingSet_SameGroupPrefix.sh LookUp_LargeSlope_Angle_17_21 LookUp_Tracking_NoFixing Clean_RollOuts_LargeSlope_20Steps_Start12_Large5_Angle_17_21_2Steps $GroupName $GroupName 0.0 

#GroupName=Group1 && bash Desktop_LookUp_TimingObj_trackExp_TrainingSet_SameGroupPrefix.sh LookUp_LargeSlope_Angle_21_26 LookUp_TimingObj_Tracking_NoFixing CleanRollOuts_LargeSlope_20Steps_Start12_Large5_Angle21_26_2Steps $GroupName $GroupName 0.0 

#---------------------
#Get the path of the Working Directory
#cd /afs/inf.ed.ac.uk/group/project/mlp_localobj/
cd /home/jiayu/Desktop/MLP_DataSet/
#mkdir $1
cd $1
pwd
filedir=$PWD

echo $filedir

compute_rounds=1

#Get the path of Training RollOuts
#rootfolder=/afs/inf.ed.ac.uk/group/project/mlp_localobj/ 
rootfolder=/home/jiayu/Desktop/MLP_DataSet/ 
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

    #python3 -W ignore /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \

    python3 /home/jiayu/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen_time_localobj.py \
    -WorkingDirectory $filedir \
    -RollOutFolderName $2 \
    -Exp_Prefix $5 \
    -LocalObj_from fromFile \
    -LocalObjTrackingFlag Yes \
    -NoisyLocalObj Yes \
    -NoiseLevel $6 \
    -InitConditionType fromFirstRoundTraj \
    -InitConditionFilePath $envfile \
    -EnvModelPath $envfile \
    -NumofRounds 10 \
    -VisualizationFlag No \

    let compute_rounds=compute_rounds+1

   if  [[ $compute_rounds -gt $numExp ]]; then
      break
   fi

done