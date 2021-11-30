#External Inputs
#$1 --- Working Direcotry
#$2 --- Folder name for saving the computed rollouts
#$3 --- Folder name for saved NN Tracking Experiments
#$4 --- Prefix of the Target Experiment File name (That will be loaded for Envrionments)
#$5 --- Prefix of the RollOut file name

#Command Example (On server):
#GroupName=Group20 && longjob -28day -c "nice bash server_re_rollout_from_every_predicted_state.sh Rubbles_DaggerExact ReRollOut_TrainingInitialSet_Dagger_1Iter NN_TrackTraining_InitSet_Dagger_1Iter $GroupName $GroupName"

#GroupName=Group20 && longjob -28day -c "nice bash server_re_rollout_from_every_predicted_state.sh Rubbles_RegretOneStep ReRollOut_TrainingInitialSet_All_Aug_1StepBeforeFail_1Time NN_TrackValidation_Aug_1StepBeforeFail_1Time $GroupName $GroupName"

#GroupName=Group20 && longjob -28day -c "nice bash server_re_rollout_from_every_predicted_state.sh Rubbles_RegretOneStep ReRollOut_TrainingInitialSet_All_InitialSet NN_TrackTraining_InitialAll $GroupName $GroupName"


#GroupName=Group20 && longjob -28day -c "nice bash server_re_rollout_from_every_predicted_state.sh LargeSlope_Angle_17_26 ReRollOut_LargeSlope_TrainingInitialSet_All_InitialSet NN_TrackTrainingAll_LargeSlope_Start12_13_Large_5_InitialSet $GroupName $GroupName"

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

    #python3 /home/jiayu/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen_rollout_from_every_predict_state.py \ 
    python3 -W ignore /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen_rollout_from_every_predict_state.py \
    -WorkingDirectory $filedir \
    -RollOutFolderName $2 \
    -Exp_Prefix $5 \
    -EnvModelPath $envfile \

    let compute_rounds=compute_rounds+1

   if  [[ $compute_rounds -gt $numExp ]]; then
      break
   fi

done