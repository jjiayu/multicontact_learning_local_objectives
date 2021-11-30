#$1 --- the firrst argument --- Defines the Name of the Working Directory (Flat/Rubbles_with_OneLargeSlope)
#$2 --- the second argument --- Defines the name of the Folder for Storing Rollouts going to be computed(i.e. RawTrainingSetRollOuts)
#$3 --- the third argument  --- Defines the name of the Folder Stores the Extracted Large Slope Setup
#$4 --- the forth argument  --- Defines the group prefix of the unseen state files that we want to compute from
#$5 --- the fifth argument  --- Defines the group prefix giving to the rollout computation
#$6 --- the sixth argument  --- Defines how many step we want to compute in a single rollout
#$7 --- the seventh argument--- Defines how many lookahead steps we want

#GroupName=Group20 && longjob -28day -c "nice bash server_rollout_for_large_slope.sh LargeSlope RollOuts_LargeSlope_20Steps_Start15_Large5 LargeSlope_Start15_Large5 $GroupName $GroupName 20 4"

#GroupName=Group20 && longjob -28day -c "nice bash server_rollout_for_large_slope.sh LargeSlope_Angle_21_26 RollOuts_LargeSlope_20Steps_Start12_Large5_Angle21_26_4Steps LargeSlope_Start12_Large_5_Angle21_26 $GroupName $GroupName 20 4"

#GroupName=Group20 && longjob -28day -c "nice bash server_rollout_for_large_slope.sh LargeSlope_Angle_22_X_negative RollOuts_LargeSlope_20Steps_Start12_Large5_Angle22_X_negative_4Steps LargeSlopeSetup_Start12_Large5_X_negative_Angle22 $GroupName $GroupName 20 4"

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
echo "Folder to get Extracted Large Slope Setup: "$PWD

sleep 7

echo "-------------------------------"

#Start Computation

#Loop over all the files
for filename in $4_*.p; do
    echo "Compute for Large Slope: "$filename
    #python3 -W ignore /home/jiayu/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \
    python3 -W ignore /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \
    -WorkingDirectory $workingDir \
    -RollOutFolderName $2 \
    -NumofRounds $6 \
    -Exp_Prefix $5 \
    -InitConditionType fromFile \
    -InitConditionFilePath $(realpath $filename) \
    -NumLookAhead $7
done






