#$1 --- the firrst argument --- Defines the Name of the Working Directory (Flat/Rubbles_with_OneLargeSlope)
#$2 --- the second argument --- Defines the name of the Folder for Storing Rollouts going to be computed(i.e. RawTrainingSetRollOuts)
#$3 --- the third argument  --- Defines the name of the Folder Stores the Extracted Unseen State (i.e. Unseen State, not with group child folder)
#$4 --- the forth argument  --- Defines the group prefix of the unseen state files that we want to compute from
#$5 --- the fifth argument  --- Defines the group prefix giving to the rollout computation
#$6 --- the sixth argument  --- Defines how many step we want to compute in a single rollout

#GroupName=Group20 && longjob -28day -c "nice bash server_rollout_compute_from_UnseenState.sh Rubbles_RegretOneStep RollOuts_from_Unseen_1StepBeforeFail_TrackTraining_Aug_1StepBeforeFail_2Time Unseen_1StepBeforeFail_TrackTrainingAll_Aug_1StepBeforeFail_2Time $GroupName $GroupName 4"

#GroupName=Group20 && longjob -28day -c "nice bash server_rollout_compute_from_UnseenState.sh Rubbles_Add2Steps_1StepbeforeFail_RemovebyClip RollOuts_4Steps_from_1StepBeforeFail_TrackTrainingAll_Add2Steps_1StepsbeforeFail_2Time_RemovebyClip Unseen_1StepBeforeFail_TrackTrainingAll_Add2Steps_1StepbeforeFail_1Time_RemovebyClip $GroupName $GroupName 4"
#GroupName=Group20 && longjob -28day -c "nice bash server_rollout_compute_from_UnseenState.sh Rubbles_Add1Step_1to2StepsbeforeFail_RemovebyClip RollOuts_4Steps_from_1StepbeforeFail_TrackTrainingAll_Add1Step_1to2StepsbeforeFail_2Time_RemovebyClip Unseen_1StepbeforeFail_TrackTrainingAll_Add1Step_1to2StepsbeforeFail_1Time_RemovebyClip $GroupName $GroupName 4"

#GroupName=Group20 && longjob -28day -c "nice bash server_rollout_compute_from_UnseenState.sh Rubbles_AddVarSteps_1to2StepbeforeFail_RemovebyClip RollOuts_20Steps_from_2StepbeforeFail_TrackTrainingAll_AddVarSteps_1to2StepbeforeFail_2Time_RemoveyClip_SmallThre Unseen_2StepBeforeFail_TrackTrainingAll_AddVarSteps_1to2StepbeforeFail_2Time_RemoveyClip_SmallThre $GroupName $GroupName 20"

#GroupName=Group20 && longjob -28day -c "nice bash server_rollout_compute_from_UnseenState.sh LargeSlope_Angle_23_X_negative RollOuts_LargeSlope_10Steps_2StepbeforeFail_TrackTrainingAll_TrainingInitial Unseen_2StepbeforeFail_LargeSlope $GroupName $GroupName 10"

#cd /afs/inf.ed.ac.uk/group/project/mlp_localobj/

#homepath=/home/jiayu/Desktop/MLP_DataSet/

homepath=/afs/inf.ed.ac.uk/group/project/mlp_localobj/

cd $homepath
echo "home path: "$homepath

#Get into the working directory
cd $1
workingDir=$PWD
echo "Current Working Directory: "$workingDir

#mkdir for the rollout folder for storing
mkdir $2
cd $2

#Save the file path
filedir=$PWD
echo "Folder to store RollOut files: "$filedir

#Go to the folder where stores the extracted unseenstate, so home foler -> work folder -> extracted unseen state folder
cd $homepath && cd $1 && cd $3
echo "Folder to get Extracted Unseen State: "$PWD

sleep 7

echo "-------------------------------"

#Start Computation

#Loop over all the files
for filename in $4_*.p; do
    echo "Compute from UnseenState: "$filename
    #python3 -W ignore /home/jiayu/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \
    python3 -W ignore /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \
    -WorkingDirectory $workingDir \
    -RollOutFolderName $2 \
    -NumofRounds $6 \
    -Exp_Prefix $5 \
    -InitConditionType fromFile \
    -InitConditionFilePath $(realpath $filename)
done






