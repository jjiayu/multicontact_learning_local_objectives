#External Inputs
#$1 --- Working Direcotry
#$2 --- Folder name for Saving the Experiments
#$3 --- Folder name for the one Saves Ground Truth Training Set
#$4 --- Prefix of the Target Experiment File name (That will be loaded for Envrionments)
#$5 --- Prefix of the Exp File name (For saved files)
#$6 --- The path of Machine learning folder (NN_Model_Valid)
#$7 --- Number of Steps we want to make (Idx +1)
#$8 --- Number of Steps with lookup table

#Command Example (On server):

#GroupName=Group1 && bash Desktop_NN_trackExp_LargeSlope_StartwithLookUp_TrainingSet_SameGroupPrefix.sh TimeTrack_LargeSlopeOnly_Angle_17_26 NN_TrackTrainingAll_LargeSlope_Start12_13_Large_5_InitialSet Clean_TrainingAll_RollOuts_LargeSlope_20Steps_Start12_13_Large5_4Steps $GroupName $GroupName /home/jiayu/Desktop/MLP_DataSet/TimeTrack_LargeSlopeOnly_Angle_17_26/ML_Models/NN_Model_AllSet 8 2

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

    #python3 /home/jiayu/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \ 
    #python3 -W ignore /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \
    python3 -W ignore /home/jiayu/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen_few_lookupobj.py \
    -WorkingDirectory $filedir \
    -RollOutFolderName $2 \
    -Exp_Prefix $5 \
    -EnvModelPath $envfile \
    -LocalObjTrackingFlag Yes \
    -LocalObj_from NeuralNetwork \
    -InitConditionType fromFirstRoundTraj \
    -InitConditionFilePath $envfile \
    -ML_ModelPath $6 \
    -NumofRounds $7 \
    -VisualizationFlag No \
    -NumSteps_Use_LookUpObj $8


    let compute_rounds=compute_rounds+1

   if  [[ $compute_rounds -gt $numExp ]]; then
      break
   fi

done