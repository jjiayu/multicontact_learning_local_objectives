#$1 --- the first argument  --- Defines the Name of the Working Directory (Flat/Rubbles_with_OneLargeSlope)
#$2 --- the second argument --- Defines the name of the Folder for Storing Rollouts going to be computed(i.e. RawTrainingSetRollOuts)
#$3 --- the third argument  --- Defines the name of the Folder Environment Model
#$4 --- the forth argument  --- Defines the group prefix of the unseen state files that we want to compute from
#$5 --- the fifth argument  --- Defines the group prefix giving to the rollout computation
#$6 --- the sixth argument  --- Defines how many step we want to compute in a single rollout
#$7 --- the seventh argument--- Defines number of lookahead
#$8 ----the eighth arugment ----VisualizationFlag

#GroupName=Group1 && bash desktop_rollout_compute_from_existing_terrain.sh Testing_Largeslope_TimeTrack_Angle_17_26 RollOut3StepNLP_AllTestScenarios CleanValidationRollOut $GroupName $GroupName 8 3 No

#cd /afs/inf.ed.ac.uk/group/project/mlp_localobj/

homepath=/home/jiayu/Desktop/MLP_DataSet/

#homepath=/afs/inf.ed.ac.uk/group/project/mlp_localobj/

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
    #python3 -W ignore /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \
    python3 -W ignore /home/jiayu/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \
    -WorkingDirectory $workingDir \
    -RollOutFolderName $2 \
    -NumofRounds $6 \
    -NumLookAhead $7 \
    -Exp_Prefix $5 \
    -EnvModelPath $(realpath $filename) \
    -VisualizationFlag $8
done






